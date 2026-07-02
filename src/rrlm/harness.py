"""Build the RLM-first agent: PredictRLM + doctrine skill + depth-gated rlm_spawn."""

from __future__ import annotations

import inspect
import os
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import dspy
from predict_rlm import File, PredictRLM, Skill

from rrlm.config import HarnessConfig
from rrlm.metrics import _inline_cost
from rrlm.pi_config import ResolvedModel
from rrlm.playbooks import doctrine_skill

# Constructor params of the installed predict-rlm, computed once. Used to
# feature-detect optional kwargs (e.g. max_action_generation_retries, which only
# exists in patched builds) so rrlm runs against stock predict-rlm from PyPI.
_PREDICT_RLM_PARAMS = set(inspect.signature(PredictRLM.__init__).parameters)


class BudgetExceededError(RuntimeError):
    """A per-run global budget ceiling (sub-LM calls or spend) was reached."""


class RunBudget:
    """Per-run ceilings shared across every recursion depth.

    predict-rlm enforces ``max_llm_calls`` per agent instance, so each
    ``rlm_spawn`` child would otherwise get a fresh allowance and the "cap"
    would multiply with the spawn tree. This object closes that hole: attached
    to the SharedLM instances (which every depth shares), it makes the sub-call
    ceiling, the spend ceiling, and the spawn ceiling global to the run.
    """

    def __init__(
        self,
        max_sub_calls: int | None = None,
        max_cost_usd: float | None = None,
        max_spawns: int | None = None,
    ):
        self.max_sub_calls = max_sub_calls
        self.max_cost_usd = max_cost_usd
        self.max_spawns = max_spawns
        self.sub_calls = 0
        self.cost_usd = 0.0
        self.spawns = 0
        self._lock = threading.Lock()

    def check_cost(self) -> None:
        """Raise once accumulated spend has crossed the ceiling.

        Soft by construction: checked before each call, so a run overshoots by
        at most the one call already in flight. Local models report no cost
        and never trip it.
        """
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"cost budget exhausted: ${self.cost_usd:.4f} spent, "
                f"max_cost_usd=${self.max_cost_usd:.4f}"
            )

    def take_sub_call(self) -> None:
        with self._lock:
            if self.max_sub_calls is not None and self.sub_calls >= self.max_sub_calls:
                raise BudgetExceededError(
                    f"global sub-LM call budget exhausted ({self.max_sub_calls} calls "
                    "made across all depths). Stop fanning out and SUBMIT the best "
                    "verified answer from what you already have."
                )
            self.sub_calls += 1

    def take_spawn(self) -> bool:
        """Reserve one rlm_spawn slot; False when the global ceiling is hit."""
        with self._lock:
            if self.max_spawns is not None and self.spawns >= self.max_spawns:
                return False
            self.spawns += 1
            return True

    def register_cost(self, cost: float | None) -> None:
        if cost:
            with self._lock:
                self.cost_usd += float(cost)


def _response_cost(response) -> float | None:
    """Best per-call cost figure available on a litellm response object."""
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and hidden.get("response_cost") is not None:
        return hidden["response_cost"]
    return _inline_cost(getattr(response, "usage", None))


class SharedLM(dspy.LM):
    """LM whose no-argument copy() returns itself, with optional run budgets.

    predict-rlm copies the lm/sub_lm instances it is given (predict_rlm.py:795),
    so with a plain dspy.LM each recursion level accumulates history on a private
    copy and per-run accounting harvests nothing. Identity-copy keeps every call
    at every depth in one .history. copy() with overrides still returns a real
    copy, so per-call config overrides keep their isolation semantics.

    When a RunBudget is attached (see ``rrlm.solve``), every call first checks
    the global spend ceiling, sub-role calls also draw from the global sub-call
    ceiling, and each response's cost is accumulated, across all depths, since
    the same instance is shared by the whole spawn tree.
    """

    _rrlm_budget: RunBudget | None = None
    _rrlm_role: str = "main"

    def copy(self, **kwargs):
        if not kwargs:
            return self
        return super().copy(**kwargs)

    def attach_budget(self, budget: RunBudget, role: str) -> None:
        self._rrlm_budget = budget
        self._rrlm_role = role

    def _budget_precheck(self) -> None:
        budget = self._rrlm_budget
        if budget is None:
            return
        budget.check_cost()
        if self._rrlm_role == "sub":
            budget.take_sub_call()

    def _budget_register(self, response) -> None:
        if self._rrlm_budget is not None:
            self._rrlm_budget.register_cost(_response_cost(response))

    def forward(self, *args, **kwargs):
        self._budget_precheck()
        response = super().forward(*args, **kwargs)
        self._budget_register(response)
        return response

    async def aforward(self, *args, **kwargs):
        self._budget_precheck()
        response = await super().aforward(*args, **kwargs)
        self._budget_register(response)
        return response


class RlmTask(dspy.Signature):
    """Solve the task by writing Python in the REPL over the `data` variable."""

    task: str = dspy.InputField(desc="What to accomplish")
    data: str = dspy.InputField(desc="The working set; explore it programmatically")
    answer: str = dspy.OutputField(desc="The final, verified answer")


class BaselineTask(dspy.Signature):
    """Answer the task using the provided data."""

    task: str = dspy.InputField()
    data: str = dspy.InputField()
    answer: str = dspy.OutputField()


_FILES_DOC = (
    " Input files are mounted into the sandbox; the `files` variable lists "
    "their paths. Read them with Python."
)


def make_signature(answer_type: type | None = None, with_files: bool = False):
    """Build the run signature: optionally typed answer and/or file inputs.

    ``answer_type`` is any type predict-rlm can parse a SUBMIT value into
    (str, int, float, bool, dict, list[...], a Pydantic model, ...). With
    ``with_files`` the signature gains a ``files: list[File]`` input that
    predict-rlm mounts into the sandbox as real files.
    """
    answer_type = answer_type or str
    if answer_type is str and not with_files:
        return RlmTask
    annotations: dict = {"task": str, "data": str}
    attrs: dict = {
        "task": dspy.InputField(desc="What to accomplish"),
        "data": dspy.InputField(desc="The working set; explore it programmatically"),
        "__doc__": RlmTask.__doc__ + (_FILES_DOC if with_files else ""),
    }
    if with_files:
        annotations["files"] = list[File]
        attrs["files"] = dspy.InputField(desc="Input files, mounted into the sandbox")
    annotations["answer"] = answer_type
    attrs["answer"] = dspy.OutputField(desc="The final, verified answer")
    attrs["__annotations__"] = annotations
    return type("RlmTaskDynamic", (dspy.Signature,), attrs)


# File extensions that benefit from a predict-rlm document skill. The skill
# bundles instructions + packages; on the jspi backend packages auto-install
# (micropip), on supervisor/sbx the host/container environment must provide
# them (document this to users rather than failing late).
_EXT_SKILL = {
    ".pdf": "pdf",
    ".xlsx": "spreadsheet",
    ".xlsm": "spreadsheet",
    ".xls": "spreadsheet",
    ".csv": "spreadsheet",
    ".docx": "docx",
}


def document_skills_for(paths: Iterable[str | Path]) -> list[Skill]:
    """The predict-rlm document skills matching the given files' extensions."""
    from predict_rlm import skills as _skills

    names = sorted(
        {
            _EXT_SKILL[ext]
            for p in paths
            if (ext := os.path.splitext(str(p))[1].lower()) in _EXT_SKILL
        }
    )
    return [getattr(_skills, name) for name in names]


def build_lm(
    model: ResolvedModel,
    max_tokens: int,
    temperature: float,
    reasoning: str = "default",
) -> dspy.LM:
    """Build a dspy.LM from a Pi-resolved model (see ``rrlm.pi_config``)."""
    if max_tokens > model.max_tokens:
        raise ValueError(
            f"max_tokens {max_tokens} exceeds {model.ref} limit {model.max_tokens}"
        )

    if model.needs_json_schema:
        # make DSPy/litellm emit response_format json_schema (LM Studio-compatible)
        # instead of json_object, which LM Studio's OpenAI endpoint rejects.
        import litellm

        litellm.register_model({model.litellm_id: {"supports_response_schema": True}})

    if reasoning not in ("default", "off", "low", "medium", "high"):
        raise ValueError(f"unknown reasoning setting: {reasoning}")

    kwargs: dict = {}
    extra_body: dict = {}
    if model.reasoning_style == "openrouter":
        if model.openrouter_routing:
            extra_body["provider"] = model.openrouter_routing
        if reasoning == "off":
            extra_body["reasoning"] = {"enabled": False}
        elif reasoning in ("low", "medium", "high"):
            extra_body["reasoning"] = {"effort": reasoning}
    elif model.reasoning_style == "chat_template":
        # local mlx_lm/vllm/llama.cpp and openai-compatible chat servers: thinking
        # is a chat-template kwarg. effort levels collapse to on/off since the
        # template only exposes a boolean. "default" sends nothing.
        if reasoning == "off":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif reasoning in ("low", "medium", "high"):
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    elif model.reasoning_style == "native":
        # Hosted first-party APIs (Anthropic, OpenAI, Gemini, ...): these reject
        # unknown body fields, so never send chat_template_kwargs. Effort levels
        # map to litellm's standardized reasoning_effort; "off" sends nothing
        # (thinking is opt-in on these APIs, so absence IS off).
        if reasoning in ("low", "medium", "high"):
            kwargs["reasoning_effort"] = reasoning

    if extra_body:
        kwargs["extra_body"] = extra_body
    if model.api_base:  # openai-compatible endpoint (local/proxy); key may be a placeholder
        kwargs["api_base"] = model.api_base
    if model.is_local:
        # local generation is far slower than hosted; the 600s litellm default
        # turns one slow turn into a 4x-retried multi-hour stall. Give locals
        # room and do not retry a timeout (it will just time out again).
        kwargs["timeout"] = 1800
        kwargs["num_retries"] = 0
    else:
        kwargs["num_retries"] = 3
    return SharedLM(
        model.litellm_id,
        api_key=model.api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=False,  # measure real calls, never cache hits
        **kwargs,
    )


def _set_sandbox_exec_timeout(seconds: float) -> None:
    """Raise the JSPI backend per-turn exec timeout (default 300s).

    PredictRLM exposes no pass-through for the jspi exec_timeout, so patch the
    constructor default. A wide local fan-out runs as one REPL turn over serial
    leaves and otherwise trips the 300s cap mid-gather. Idempotent.

    predict-rlm 0.7 renamed the class JspiInterpreter -> JspiBackend and moved it
    to backends.jspi.backend.
    """
    from predict_rlm.backends.jspi.backend import JspiBackend

    if getattr(JspiBackend, "_rrlm_exec_timeout", None) == seconds:
        return
    base_init = getattr(JspiBackend, "_rrlm_base_init", JspiBackend.__init__)

    def patched_init(self, *args, exec_timeout=seconds, **kwargs):
        base_init(self, *args, exec_timeout=exec_timeout, **kwargs)

    JspiBackend._rrlm_base_init = base_init
    JspiBackend.__init__ = patched_init
    JspiBackend._rrlm_exec_timeout = seconds


def build_rlm(
    cfg: HarnessConfig,
    main_lm: dspy.LM,
    sub_lm: dspy.LM,
    *,
    depth: int = 0,
    spawn_stats: Counter | None = None,
    signature=None,
    budget: RunBudget | None = None,
    extra_tools: list[Callable] | None = None,
    extra_skills: list[Skill] | None = None,
    doctrine: str | None = None,
) -> PredictRLM:
    """Construct the agent for one depth level.

    LM instances are shared across all depths so accounting stays centralized
    in their .history. `spawn_stats` counts rlm_spawn invocations per depth.
    ``signature`` (root only; children use the default) supports typed answers
    and file inputs, ``extra_tools``/``extra_skills`` are caller-provided
    host-side tools and predict-rlm Skill bundles, and ``doctrine`` overrides
    the built-in doctrine text (the RLM-GEPA optimization target).
    """
    spawn_stats = spawn_stats if spawn_stats is not None else Counter()
    tools: list[Callable] = []
    if depth < cfg.max_depth:
        tools.append(
            _make_rlm_spawn(
                cfg, main_lm, sub_lm, depth, spawn_stats,
                budget=budget, extra_tools=extra_tools,
                extra_skills=extra_skills, doctrine=doctrine,
            )
        )
    if extra_tools:
        tools.extend(extra_tools)

    skills = [doctrine_skill(doctrine)]
    if cfg.web:
        # Host-side web tools work in every backend (predict-rlm bridges tool
        # calls to the host). The web doctrine tells the agent to retrieve and
        # verify instead of answering from memory.
        from rrlm.playbooks import web_skill
        from rrlm.webtools import web_tools

        tools.extend(web_tools())
        skills.append(web_skill())
    if extra_skills:
        skills.extend(extra_skills)

    # Backend selection. "supervisor" (0.7) runs real local CPython with no
    # Deno/WASM bridge (fastest for wide local fan-out and no JSPI hang) and
    # is passed as interpreter= (mutually exclusive with sandbox_backend).
    rlm_kwargs: dict = dict(
        lm=main_lm,
        sub_lm=sub_lm,
        skills=skills,
        tools=tools,
        max_iterations=cfg.max_iterations,
        max_llm_calls=cfg.max_llm_calls,
    )
    # Stock predict-rlm (PyPI) has no per-turn action retry; only pass it when the
    # installed build supports it (e.g. a patched/forked predict-rlm).
    if "max_action_generation_retries" in _PREDICT_RLM_PARAMS:
        rlm_kwargs["max_action_generation_retries"] = cfg.max_action_retries
    if cfg.backend == "supervisor":
        from predict_rlm import DirectPythonBackend

        rlm_kwargs["interpreter"] = DirectPythonBackend(
            exec_timeout=cfg.sandbox_exec_timeout
        )
    elif cfg.backend == "sbx":
        from predict_rlm import SbxConfig

        sbx_kwargs: dict = {"exec_timeout": cfg.sandbox_exec_timeout}
        # RRLM_SBX_NAME -> a persistent, reused container (reuse=True implies
        # persist=True). It survives process exit, so successive rrlm-solve CLI
        # calls reuse one warm sandbox and skip the ~25s per-call create cost.
        name = os.environ.get("RRLM_SBX_NAME")
        if name:
            sbx_kwargs.update(name=name, reuse=True)
        rlm_kwargs["sandbox_backend"] = "sbx"
        rlm_kwargs["sbx_config"] = SbxConfig(**sbx_kwargs)
    else:  # jspi
        if cfg.sandbox_exec_timeout != 300.0:
            _set_sandbox_exec_timeout(cfg.sandbox_exec_timeout)
        rlm_kwargs["sandbox_backend"] = "jspi"

    rlm = PredictRLM(signature or RlmTask, **rlm_kwargs)
    rlm.spawn_stats = spawn_stats  # surfaced in the run result
    return rlm


def _make_rlm_spawn(
    cfg: HarnessConfig,
    main_lm: dspy.LM,
    sub_lm: dspy.LM,
    depth: int,
    spawn_stats: Counter,
    *,
    budget: RunBudget | None = None,
    extra_tools: list[Callable] | None = None,
    extra_skills: list[Skill] | None = None,
    doctrine: str | None = None,
):
    child_depth = depth + 1

    async def rlm_spawn(task: str, data: str) -> str:
        """Spawn a child recursive agent over a large data slice.

        Use ONLY when a sub-problem's working set is too large to handle with a
        few predict() calls in this REPL (capacity-driven recursion). The child
        gets its own REPL with the same tools. Returns the child's final answer
        as a string. Prefer breadth (parallel predict calls) over depth.
        """
        if budget is not None and not budget.take_spawn():
            return (
                "rlm_spawn refused: the global spawn budget for this run is "
                "exhausted. Do the remaining work in this REPL with code and "
                "predict() calls, then SUBMIT."
            )
        spawn_stats[child_depth] += 1
        child = build_rlm(
            cfg, main_lm, sub_lm, depth=child_depth, spawn_stats=spawn_stats,
            budget=budget, extra_tools=extra_tools, extra_skills=extra_skills,
            doctrine=doctrine,
        )
        prediction = await child.acall(task=task, data=data)
        return prediction.answer

    return rlm_spawn
