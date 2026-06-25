"""Harness construction tests -- no network calls."""

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
