"""Tests for rrlm.gepa: dataset loading, checkers, and the doctrine project.

The evaluate_example test is a real integration run: the project executes the
actual rrlm harness against the offline stub server (supervisor backend), and
the checker scores the real answer. Requires the rlm_gepa package (skipped when
the [gepa] extra is not installed).
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from rrlm.gepa import DatasetExample, load_dataset, score_answer, split_dataset
from rrlm.playbooks import DOCTRINE


# --- dataset loading -------------------------------------------------------- #
def _write_dataset(tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_load_dataset_inline_and_file(tmp_path):
    (tmp_path / "payload.txt").write_text("row1\nrow2\n", encoding="utf-8")
    path = _write_dataset(tmp_path, [
        {"instruction": "sum it", "data": "1 2 3", "expected": "6", "checker": "number"},
        {"instruction": "find it", "data_file": "payload.txt", "expected": "row2"},
    ])
    examples = load_dataset(path)
    assert len(examples) == 2
    assert examples[0].checker == "number"
    assert examples[1].data == "row1\nrow2\n"  # data_file resolved and read
    assert examples[1].checker == "exact"  # the default


def test_load_dataset_rejects_bad_rows(tmp_path):
    with pytest.raises(ValueError, match="required"):
        load_dataset(_write_dataset(tmp_path, [{"instruction": "", "expected": "x"}]))
    with pytest.raises(ValueError, match="unknown checker"):
        load_dataset(_write_dataset(
            tmp_path, [{"instruction": "q", "expected": "x", "checker": "vibes"}]
        ))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(empty)


def test_split_dataset_honors_explicit_and_auto():
    ex = [
        DatasetExample(f"e{i}", "q", "d", "x", split=s)
        for i, s in enumerate(["train", "val", "", "", "", "", ""])
    ]
    train, val = split_dataset(ex)
    assert {e.example_id for e in val} >= {"e1"}  # explicit val kept
    assert {e.example_id for e in train} >= {"e0"}  # explicit train kept
    assert len(train) + len(val) == len(ex)
    assert len(val) >= 2  # explicit + every-4th auto assignment


def test_split_dataset_requires_both_splits():
    ex = [DatasetExample("e0", "q", "d", "x", split="train")]
    with pytest.raises(ValueError, match="both splits"):
        split_dataset(ex)


# --- checkers ---------------------------------------------------------------- #
def test_score_answer_exact():
    assert score_answer(" 42 ", "42", "exact")[0] == 1.0
    score, feedback = score_answer("41", "42", "exact")
    assert score == 0.0 and "expected exactly" in feedback


def test_score_answer_contains():
    assert score_answer("The id is P203.", "p203", "contains")[0] == 1.0
    assert score_answer("no match here", "P203", "contains")[0] == 0.0


def test_score_answer_number():
    assert score_answer("Total: 1,234.50 USD", "1234.50", "number")[0] == 1.0
    score, feedback = score_answer("1234.49", "1234.50", "number")
    assert score == 0.0 and "off by" in feedback
    score, feedback = score_answer("none found", "10", "number")
    assert score == 0.0 and "no numeric value" in feedback


# --- the doctrine project (needs rlm_gepa) ----------------------------------- #
rlm_gepa = pytest.importorskip("rlm_gepa")


def _project(tmp_path, model_ref):
    from rrlm.gepa import build_project

    data = "alpha\nbeta\n"
    rows = [
        {"instruction": f"q{i}", "data": data, "expected": str(len(data))}
        for i in range(8)
    ]
    path = _write_dataset(tmp_path, rows)
    return build_project(path, main_model=model_ref, sub_model=model_ref), data


def test_build_project_requires_dataset(monkeypatch):
    from rrlm.gepa import build_project

    monkeypatch.delenv("RRLM_GEPA_DATASET", raising=False)
    with pytest.raises(ValueError, match="no dataset"):
        build_project(None)


@pytest.mark.integration
def test_build_project_seed_spec_and_splits(stub_model, tmp_path):
    model = stub_model("submit")
    project, _ = _project(tmp_path, model)
    assert project.components == ("doctrine",)
    assert project.seed_candidate() == {"doctrine": DOCTRINE}
    assert len(project.load_trainset()) + len(project.load_valset()) == 8
    # the AgentSpec is derived from a real constructed harness
    assert "rlm_spawn" in project.agent_spec.tool_signatures
    assert project.agent_spec.target_signature


def _mk_context(**overrides):
    """Build an EvaluationContext across rlm_gepa versions: fill required
    params with sensible offline defaults."""
    from rlm_gepa import EvaluationContext

    defaults = {
        "lm": None, "sub_lm": None, "max_iterations": 5, "task_timeout": 60,
        "output_dir": None, "kind": "train", "verbose_rlm": False,
        "debug_rlm": False, "concurrency": 1, "telemetry_context": None,
        "task_resources": None,
    }
    params = inspect.signature(EvaluationContext).parameters
    kwargs = {name: defaults.get(name) for name, p in params.items()
              if p.default is inspect.Parameter.empty}
    kwargs.update({k: v for k, v in defaults.items() if k in params})
    kwargs.update(overrides)
    return EvaluationContext(**kwargs)


@pytest.mark.integration
def test_evaluate_example_scores_a_real_run(stub_model, tmp_path):
    """The GEPA evaluator runs the REAL harness (stub server, supervisor
    backend, real REPL execution) and the checker scores the real answer."""
    model = stub_model("submit")
    project, data = _project(tmp_path, model)
    example = project.load_trainset()[0]
    result = asyncio.run(
        project.evaluate_example(project.seed_candidate(), example, _mk_context())
    )
    assert result.score == 1.0  # the stub submits str(len(data)) == expected
    assert result.example_id == example.example_id
    assert result.traces  # the RunTrace was captured for the proposer
    assert result.error is None


@pytest.mark.integration
def test_evaluate_example_feedback_on_wrong_answer(stub_model, tmp_path):
    from rrlm.gepa import build_project

    model = stub_model("submit")
    path = _write_dataset(tmp_path, [
        {"instruction": f"q{i}", "data": "abc", "expected": "999"} for i in range(8)
    ])
    project = build_project(path, main_model=model, sub_model=model)
    example = project.load_trainset()[0]
    result = asyncio.run(
        project.evaluate_example(project.seed_candidate(), example, _mk_context())
    )
    assert result.score == 0.0
    assert "expected exactly" in result.feedback
