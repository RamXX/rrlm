"""Build the RLM-first agent: PredictRLM + doctrine skill + depth-gated rlm_spawn."""

from __future__ import annotations

import inspect
from collections import Counter

import dspy
from predict_rlm import PredictRLM

from rrlm.config import HarnessConfig
from rrlm.pi_config import ResolvedModel
from rrlm.playbooks import doctrine_skill

# Constructor params of the installed predict-rlm, computed once. Used to
# feature-detect optional kwargs (e.g. max_action_generation_retries, which only
# exists in patched builds) so rrlm runs against stock predict-rlm from PyPI.
_PREDICT_RLM_PARAMS = set(inspect.signature(PredictRLM.__init__).parameters)


class SharedLM(dspy.LM):
    """LM whose no-argument copy() returns itself.

    predict-rlm copies the lm/sub_lm instances it is given (predict_rlm.py:795),
    so with a plain dspy.LM each recursion level accumulates history on a private
    copy and per-run accounting harvests nothing. Identity-copy keeps every call
    at every depth in one .history. copy() with overrides still returns a real
    copy, so per-call config overrides keep their isolation semantics.
    """

    def copy(self, **kwargs):
        if not kwargs:
            return self
        return super().copy(**kwargs)


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

    kwargs: dict = {}
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
) -> PredictRLM:
    """Construct the agent for one depth level.

    LM instances are shared across all depths so accounting stays centralized
    in their .history. `spawn_stats` counts rlm_spawn invocations per depth.
    """
    spawn_stats = spawn_stats if spawn_stats is not None else Counter()
    tools = []
    if depth < cfg.max_depth:
        tools.append(_make_rlm_spawn(cfg, main_lm, sub_lm, depth, spawn_stats))

    # Backend selection. "supervisor" (0.7) runs real local CPython with no
    # Deno/WASM bridge (fastest for wide local fan-out and no JSPI hang) and
    # is passed as interpreter= (mutually exclusive with sandbox_backend).
    rlm_kwargs: dict = dict(
        lm=main_lm,
        sub_lm=sub_lm,
        skills=[doctrine_skill()],
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
        import os

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

    rlm = PredictRLM(RlmTask, **rlm_kwargs)
    rlm.spawn_stats = spawn_stats  # surfaced in the run result
    return rlm


def _make_rlm_spawn(
    cfg: HarnessConfig,
    main_lm: dspy.LM,
    sub_lm: dspy.LM,
    depth: int,
    spawn_stats: Counter,
):
    child_depth = depth + 1

    async def rlm_spawn(task: str, data: str) -> str:
        """Spawn a child recursive agent over a large data slice.

        Use ONLY when a sub-problem's working set is too large to handle with a
        few predict() calls in this REPL (capacity-driven recursion). The child
        gets its own REPL with the same tools. Returns the child's final answer
        as a string. Prefer breadth (parallel predict calls) over depth.
        """
        spawn_stats[child_depth] += 1
        child = build_rlm(
            cfg, main_lm, sub_lm, depth=child_depth, spawn_stats=spawn_stats
        )
        prediction = await child.acall(task=task, data=data)
        return prediction.answer

    return rlm_spawn
