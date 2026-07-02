"""Harness construction tests, no network calls."""

from collections import Counter

import pytest

from rrlm.config import HarnessConfig
from rrlm.harness import build_lm, build_rlm
from rrlm.pi_config import ResolvedModel


def _model(max_tokens: int = 65_536, **kw) -> ResolvedModel:
    return ResolvedModel(
        ref="test/model",
        provider="openrouter",
        model_id="test/model",
        litellm_id="openrouter/test/model",
        api_key="sk-or-test",
        max_tokens=max_tokens,
        reasoning_style="openrouter",
        **kw,
    )


def _lms():
    cfg = HarnessConfig()
    main = build_lm(_model(), cfg.main_max_tokens, cfg.temperature)
    sub = build_lm(_model(), cfg.sub_max_tokens, cfg.temperature)
    return cfg, main, sub


def test_shared_lm_noarg_copy_is_identity():
    _, main, _ = _lms()
    assert main.copy() is main


def test_shared_lm_copy_with_overrides_is_new_instance():
    _, main, _ = _lms()
    clone = main.copy(temperature=0.9)
    assert clone is not main
    assert clone.kwargs["temperature"] == 0.9


def test_build_lm_rejects_excess_max_tokens():
    with pytest.raises(ValueError, match="exceeds"):
        build_lm(_model(max_tokens=65_536), 100_000, 0.2)


def test_build_lm_disables_cache():
    _, main, _ = _lms()
    assert main.cache is False


def test_build_rlm_exposes_spawn_below_max_depth():
    cfg, main, sub = _lms()
    rlm = build_rlm(cfg, main, sub, depth=0)
    tool_names = set(rlm.tools.keys()) if isinstance(rlm.tools, dict) else {
        t.__name__ for t in rlm.tools
    }
    assert any("rlm_spawn" in name for name in tool_names)


def test_build_rlm_omits_spawn_at_max_depth():
    cfg, main, sub = _lms()
    rlm = build_rlm(cfg, main, sub, depth=cfg.max_depth)
    tool_names = set(rlm.tools.keys()) if isinstance(rlm.tools, dict) else {
        t.__name__ for t in rlm.tools
    }
    assert not any("rlm_spawn" in name for name in tool_names)


def test_set_sandbox_exec_timeout_is_idempotent_and_applies():
    import inspect

    from predict_rlm.backends.jspi.backend import JspiBackend

    from rrlm.harness import _set_sandbox_exec_timeout

    original = JspiBackend.__init__
    try:
        _set_sandbox_exec_timeout(1234.0)
        assert JspiBackend._rrlm_exec_timeout == 1234.0
        patched = JspiBackend.__init__
        _set_sandbox_exec_timeout(1234.0)  # idempotent: no re-wrap
        assert JspiBackend.__init__ is patched
        # the patched default is visible on the signature
        assert (
            inspect.signature(JspiBackend.__init__).parameters["exec_timeout"].default
            == 1234.0
        )
    finally:
        JspiBackend.__init__ = original
        for attr in ("_rrlm_exec_timeout", "_rrlm_base_init"):
            if hasattr(JspiBackend, attr):
                delattr(JspiBackend, attr)


def test_spawn_stats_shared_counter():
    cfg, main, sub = _lms()
    stats = Counter()
    rlm = build_rlm(cfg, main, sub, depth=0, spawn_stats=stats)
    assert rlm.spawn_stats is stats


def test_max_llm_calls_threads_into_real_predict_rlm():
    cfg = HarnessConfig(max_llm_calls=7, max_iterations=9)
    main = build_lm(_model(), cfg.main_max_tokens, cfg.temperature)
    rlm = build_rlm(cfg, main, main)
    assert rlm.max_llm_calls == 7
    assert rlm.max_iterations == 9


# --- build_lm branch coverage (reasoning styles, local vs hosted, api_base) --- #
def test_build_lm_openrouter_reasoning_off():
    m = _model(supports_reasoning=True)
    lm = build_lm(m, 1024, 0.2, reasoning="off")
    assert lm.kwargs["extra_body"]["reasoning"] == {"enabled": False}


def test_build_lm_openrouter_reasoning_effort_and_routing():
    m = _model(supports_reasoning=True, openrouter_routing={"order": ["fireworks"]})
    lm = build_lm(m, 1024, 0.2, reasoning="medium")
    assert lm.kwargs["extra_body"]["reasoning"] == {"effort": "medium"}
    assert lm.kwargs["extra_body"]["provider"] == {"order": ["fireworks"]}


def test_build_lm_chat_template_thinking_toggle():
    base = dict(
        ref="local/x", provider="omlx", model_id="x", litellm_id="openai/x",
        api_key="not-needed", api_base="http://127.0.0.1:8004/v1", is_local=True,
        reasoning_style="chat_template", max_tokens=65_536,
    )
    off = build_lm(ResolvedModel(**base), 1024, 0.2, reasoning="off")
    assert off.kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    on = build_lm(ResolvedModel(**base), 1024, 0.2, reasoning="high")
    assert on.kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_build_lm_local_sets_long_timeout_no_retries():
    m = ResolvedModel(
        ref="local/x", provider="omlx", model_id="x", litellm_id="openai/x",
        api_key="not-needed", api_base="http://127.0.0.1:8004/v1", is_local=True,
        reasoning_style="chat_template", max_tokens=65_536,
    )
    lm = build_lm(m, 1024, 0.2, reasoning="default")
    assert lm.kwargs["timeout"] == 1800
    assert lm.num_retries == 0
    assert lm.kwargs["api_base"] == "http://127.0.0.1:8004/v1"


def test_build_lm_hosted_uses_retries():
    lm = build_lm(_model(), 1024, 0.2)
    assert lm.num_retries == 3


def test_build_lm_needs_json_schema_registers_model():
    import litellm

    m = ResolvedModel(
        ref="lmstudio/x", provider="lmstudio", model_id="x", litellm_id="openai/x-jsonschema",
        api_key="lm-studio", api_base="http://localhost:1234/v1", is_local=True,
        reasoning_style="chat_template", needs_json_schema=True, max_tokens=65_536,
    )
    build_lm(m, 1024, 0.2)
    # register_model recorded the schema-support flag for this litellm id.
    assert litellm.supports_response_schema(model="openai/x-jsonschema") is True


def test_build_lm_rejects_unknown_reasoning():
    with pytest.raises(ValueError, match="unknown reasoning"):
        build_lm(_model(), 1024, 0.2, reasoning="ultra")


# --- backend selection in build_rlm --- #
def test_build_rlm_supervisor_backend_constructs():
    cfg = HarnessConfig(backend="supervisor")
    main = build_lm(_model(), cfg.main_max_tokens, cfg.temperature)
    rlm = build_rlm(cfg, main, main)
    assert rlm is not None


def test_build_rlm_sbx_backend_constructs(monkeypatch):
    monkeypatch.setenv("RRLM_SBX_NAME", "rrlm-test-warm")
    cfg = HarnessConfig(backend="sbx")
    main = build_lm(_model(), cfg.main_max_tokens, cfg.temperature)
    # Constructing the agent with the sbx backend must not require Docker (no run).
    rlm = build_rlm(cfg, main, main)
    assert rlm is not None


# --- native reasoning style (hosted first-party APIs) --- #
def _native_model(**kw) -> ResolvedModel:
    return ResolvedModel(
        ref="anthropic/claude-x", provider="anthropic", model_id="claude-x",
        litellm_id="anthropic/claude-x", api_key="sk-ant-test",
        max_tokens=65_536, reasoning_style="native", supports_reasoning=True, **kw,
    )


def test_build_lm_native_off_sends_no_extra_fields():
    """The default path for Anthropic/OpenAI/Gemini: reasoning=off must add
    NOTHING to the request (these APIs reject unknown body fields)."""
    lm = build_lm(_native_model(), 1024, 0.2, reasoning="off")
    assert "extra_body" not in lm.kwargs
    assert "reasoning_effort" not in lm.kwargs


def test_build_lm_native_effort_maps_to_reasoning_effort():
    lm = build_lm(_native_model(), 1024, 0.2, reasoning="medium")
    assert lm.kwargs["reasoning_effort"] == "medium"
    assert "extra_body" not in lm.kwargs  # never chat_template_kwargs for native


def test_build_lm_native_default_sends_nothing():
    lm = build_lm(_native_model(), 1024, 0.2, reasoning="default")
    assert "extra_body" not in lm.kwargs
    assert "reasoning_effort" not in lm.kwargs


# --- RunBudget: the global per-run ceilings --- #
def test_run_budget_sub_call_ceiling():
    from rrlm.harness import BudgetExceededError, RunBudget

    budget = RunBudget(max_sub_calls=2)
    budget.take_sub_call()
    budget.take_sub_call()
    with pytest.raises(BudgetExceededError, match="sub-LM call budget"):
        budget.take_sub_call()


def test_run_budget_cost_ceiling_and_registration():
    from rrlm.harness import BudgetExceededError, RunBudget

    budget = RunBudget(max_cost_usd=0.01)
    budget.check_cost()  # under the ceiling: fine
    budget.register_cost(0.005)
    budget.register_cost(None)  # unknown cost never counts
    budget.check_cost()
    budget.register_cost(0.006)
    with pytest.raises(BudgetExceededError, match="cost budget"):
        budget.check_cost()


def test_run_budget_spawn_ceiling():
    from rrlm.harness import RunBudget

    budget = RunBudget(max_spawns=1)
    assert budget.take_spawn() is True
    assert budget.take_spawn() is False  # refusal, not an exception


def test_run_budget_unlimited_by_default():
    from rrlm.harness import RunBudget

    budget = RunBudget()
    for _ in range(100):
        budget.take_sub_call()
    assert budget.take_spawn() is True
    budget.check_cost()


def test_shared_lm_sub_budget_blocks_before_any_network():
    """A sub-role LM over budget raises BEFORE the HTTP call (offline-provable)."""
    from rrlm.harness import BudgetExceededError, RunBudget

    lm = build_lm(_model(), 1024, 0.2)
    lm.attach_budget(RunBudget(max_sub_calls=0), role="sub")
    with pytest.raises(BudgetExceededError, match="sub-LM call budget"):
        lm("hi")


def test_shared_lm_cost_budget_blocks_main_too():
    from rrlm.harness import BudgetExceededError, RunBudget

    budget = RunBudget(max_cost_usd=0.01)
    budget.register_cost(0.02)  # ceiling already crossed
    lm = build_lm(_model(), 1024, 0.2)
    lm.attach_budget(budget, role="main")
    with pytest.raises(BudgetExceededError, match="cost budget"):
        lm("hi")


def test_response_cost_extraction_precedence():
    import types

    from rrlm.harness import _response_cost

    hidden = types.SimpleNamespace(_hidden_params={"response_cost": 0.005}, usage=None)
    assert _response_cost(hidden) == 0.005
    inline = types.SimpleNamespace(_hidden_params={}, usage={"cost": 0.003})
    assert _response_cost(inline) == 0.003
    none = types.SimpleNamespace(_hidden_params={}, usage={"prompt_tokens": 5})
    assert _response_cost(none) is None


def test_rlm_spawn_refuses_over_spawn_budget():
    import asyncio

    from rrlm.harness import RunBudget, _make_rlm_spawn

    cfg, main, sub = _lms()
    stats = Counter()
    spawn = _make_rlm_spawn(cfg, main, sub, 0, stats, budget=RunBudget(max_spawns=0))
    out = asyncio.run(spawn("subtask", "slice"))
    assert "rlm_spawn refused" in out
    assert stats == Counter()  # nothing spawned, nothing counted


# --- make_signature: typed answers and file inputs --- #
def test_make_signature_default_is_shared_rlmtask():
    from rrlm.harness import RlmTask, make_signature

    assert make_signature() is RlmTask
    assert make_signature(str) is RlmTask


def test_make_signature_typed_answer():
    from rrlm.harness import make_signature

    sig = make_signature(int)
    assert sig.model_fields["answer"].annotation is int
    assert "files" not in sig.model_fields


def test_make_signature_with_files():
    from predict_rlm import File

    from rrlm.harness import make_signature

    sig = make_signature(with_files=True)
    assert sig.model_fields["files"].annotation == list[File]
    assert sig.model_fields["answer"].annotation is str
    assert "files" in (sig.__doc__ or "") or "files" in sig.instructions


def test_document_skills_for_matches_extensions():
    from predict_rlm import Skill

    from rrlm.harness import document_skills_for

    skills = document_skills_for(["a.PDF", "b.xlsx", "c.txt", "d.csv"])
    assert all(isinstance(s, Skill) for s in skills)
    assert len(skills) == 2  # pdf + spreadsheet (csv and xlsx dedupe to one)
    assert document_skills_for(["notes.txt", "code.py"]) == []


# --- extra tools / doctrine passthrough --- #
def test_build_rlm_extra_tools_registered():
    async def lookup_ticket(ticket_id: str) -> str:
        """Fetch one ticket by id."""
        return ticket_id

    cfg, main, sub = _lms()
    rlm = build_rlm(cfg, main, sub, extra_tools=[lookup_ticket])
    names = set(rlm.tools.keys()) if isinstance(rlm.tools, dict) else {
        t.__name__ for t in rlm.tools
    }
    assert any("lookup_ticket" in n for n in names)


def test_doctrine_skill_override():
    from rrlm.playbooks import DOCTRINE, doctrine_skill

    assert doctrine_skill().instructions == DOCTRINE
    assert doctrine_skill("CUSTOM DOCTRINE").instructions == "CUSTOM DOCTRINE"


def test_build_rlm_with_doctrine_and_extra_skills_constructs():
    from predict_rlm import Skill

    cfg, main, sub = _lms()
    extra = Skill(name="domain-notes", instructions="Domain facts.")
    rlm = build_rlm(cfg, main, sub, doctrine="Be terse.", extra_skills=[extra])
    assert rlm is not None
