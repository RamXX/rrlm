"""RLM-GEPA wiring: optimize the rrlm doctrine text from scored examples.

This is the missing last mile of the trace pipeline: ``RRLM_TRACE_DIR`` captures
RunTraces, and this module turns a curated dataset of (instruction, data,
expected) examples into an optimization run that evolves the doctrine text
(``rrlm.playbooks.DOCTRINE``) with the RLM-GEPA engine shipped in
``predict-rlm[gepa]``. Deploy a winner with ``rrlm-solve --doctrine <file>``.

Dataset format: a JSONL file, one example per line::

    {"instruction": "Total of ok transactions?", "data": "...", "expected": "42.17", "checker": "number"}
    {"instruction": "Which product id ...?", "data_file": "reviews.txt", "expected": "P203"}
    {"instruction": "...", "data": "...", "expected": "P2", "checker": "contains", "split": "val"}

Fields: ``instruction`` and ``expected`` are required; ``data`` (inline) or
``data_file`` (relative to the dataset file) carry the payload; ``checker`` is
``exact`` (default, stripped string equality), ``contains``, or ``number``
(first numeric token, small relative tolerance); optional ``split`` pins an
example to ``train`` or ``val`` (default: every fourth example goes to val);
optional ``domain`` tags the task family (``tabular``, ``code``, ``docs``,
...) for generalization measurement.

Usage (needs the ``gepa`` extra: ``uv sync --extra gepa``)::

    export RRLM_GEPA_DATASET=examples.jsonl RRLM_MAIN=... RRLM_SUB=...
    rrlm-gepa optimize --check
    rrlm-gepa optimize --max-metric-calls 400 --minibatch-size 8 --concurrency 4
    rrlm-gepa stats runs/<run-dir>

Generalization protocol (see docs/EVALS.md): tag every example with a
``domain``, hold some domains out of optimization entirely
(``RRLM_GEPA_HOLDOUT_DOMAINS=code,web``: they never enter train or val), and
score the winner on exactly those domains afterwards::

    RRLM_GEPA_HOLDOUT_DOMAINS=code,web rrlm-gepa optimize --max-metric-calls 400
    RRLM_GEPA_HOLDOUT_DOMAINS=code,web rrlm-gepa eval --doctrine <winner.txt>

``rrlm-gepa eval`` scores with the same checkers optimization used, so the
in-domain and held-out numbers are directly comparable; a winner that beats
the built-in doctrine in-domain but not held-out has overfit its task shapes.

The executor runs the real rrlm harness (models from your Pi config via
``RRLM_MAIN``/``RRLM_SUB``); the reflective proposer is an rlm_gepa-internal
PredictRLM whose model is set with ``--proposer-lm`` (a litellm id you have
credentials for).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from rrlm.playbooks import DOCTRINE

_NEEDS_GEPA = (
    "rrlm-gepa needs the optional 'gepa' extra. Install with "
    "`uv sync --extra gepa` (checkout) or `uv tool install 'rrlm[gepa]'`."
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class DatasetExample:
    example_id: str
    instruction: str
    data: str
    expected: str
    checker: str = "exact"
    split: str = ""  # "", "train", or "val"
    domain: str = ""  # free-form task family tag, e.g. "tabular", "code", "docs"


def load_dataset(path: str | Path) -> list[DatasetExample]:
    """Load and validate the JSONL dataset; data_file paths resolve relative
    to the dataset file so datasets are relocatable."""
    path = Path(path)
    examples: list[DatasetExample] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        instruction = rec.get("instruction", "").strip()
        expected = str(rec.get("expected", "")).strip()
        if not instruction or not expected:
            raise ValueError(f"{path}:{lineno}: 'instruction' and 'expected' are required")
        data = rec.get("data", "")
        if not data and rec.get("data_file"):
            data = (path.parent / rec["data_file"]).read_text(encoding="utf-8")
        checker = rec.get("checker", "exact")
        if checker not in ("exact", "contains", "number"):
            raise ValueError(f"{path}:{lineno}: unknown checker {checker!r}")
        examples.append(
            DatasetExample(
                example_id=rec.get("id", f"ex-{lineno:04d}"),
                instruction=instruction,
                data=data,
                expected=expected,
                checker=checker,
                split=rec.get("split", ""),
                domain=str(rec.get("domain", "")).strip(),
            )
        )
    if not examples:
        raise ValueError(f"{path}: dataset is empty")
    return examples


def split_dataset(
    examples: list[DatasetExample],
) -> tuple[list[DatasetExample], list[DatasetExample]]:
    """Honor explicit ``split`` fields; auto-assign every 4th example to val."""
    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "val"]
    auto = [e for e in examples if not e.split]
    for i, e in enumerate(auto):
        (val if i % 4 == 3 else train).append(e)
    if not train or not val:
        raise ValueError(
            f"dataset must yield both splits (got {len(train)} train / {len(val)} val); "
            "add more examples or explicit 'split' fields"
        )
    return train, val


def partition_domains(
    examples: list[DatasetExample], holdout_domains: tuple[str, ...] | list[str]
) -> tuple[list[DatasetExample], list[DatasetExample]]:
    """Split examples into (in-domain, held-out) by their ``domain`` tag.

    The held-out side never enters train or val: doctrine optimization must
    not see it, or the final report measures memorization, not
    generalization. Naming a domain no example carries is a config error and
    fails loudly (a typo would silently hold out nothing).
    """
    holdout = {d.strip() for d in holdout_domains if d.strip()}
    if not holdout:
        return list(examples), []
    known = {e.domain for e in examples if e.domain}
    missing = sorted(holdout - known)
    if missing:
        known_list = ", ".join(sorted(known)) or "none (no example has a domain tag)"
        raise ValueError(
            f"holdout domain(s) not present in the dataset: {', '.join(missing)} "
            f"(domains found: {known_list})"
        )
    in_domain = [e for e in examples if e.domain not in holdout]
    held_out = [e for e in examples if e.domain in holdout]
    if not in_domain:
        raise ValueError("every example is held out; nothing is left to optimize on")
    return in_domain, held_out


def score_answer(answer: str, expected: str, checker: str) -> tuple[float, str]:
    """Machine-check an answer: (score in {0.0, 1.0}, feedback for the proposer)."""
    got = (answer or "").strip()
    want = expected.strip()
    if checker == "contains":
        if want.lower() in got.lower():
            return 1.0, "correct: expected value present in the answer"
        return 0.0, f"wrong: expected the answer to contain {want!r}, got {got[:200]!r}"
    if checker == "number":
        want_n = float(want)
        m = _NUM.search(got.replace(",", ""))
        if m:
            got_n = float(m.group())
            tol = max(1e-6, abs(want_n) * 1e-6)
            if abs(got_n - want_n) <= tol:
                return 1.0, "correct: numeric answer matches"
            return 0.0, f"wrong: expected {want_n}, got {got_n} (off by {got_n - want_n:+g})"
        return 0.0, f"wrong: no numeric value found in {got[:200]!r}, expected {want_n}"
    # exact
    if got == want:
        return 1.0, "correct: exact match"
    return 0.0, f"wrong: expected exactly {want!r}, got {got[:200]!r}"


async def score_example_with_doctrine(
    example: DatasetExample,
    doctrine_text: str,
    *,
    main_model: str | None,
    sub_model: str | None,
    backend: str | None,
    max_iterations: int = 30,
    timeout_s: float | None = None,
) -> tuple[float, str, dict]:
    """Run one example under a doctrine; return (score, feedback, raw result).

    The single scoring path shared by GEPA optimization and ``rrlm-gepa
    eval``: whatever the optimizer scored, the held-out report scores the
    same way, so the two numbers are comparable by construction.
    """
    from rrlm.solve import asolve

    result = await asolve(
        example.instruction,
        example.data,
        main_model=main_model,
        sub_model=sub_model,
        backend=backend,
        max_iterations=max_iterations,
        timeout_s=timeout_s,
        doctrine=doctrine_text,
        reconcile_cost=False,  # callers track cost themselves; skip per-run HTTP
        return_trace=True,
    )
    error = result["error"]
    if error:
        return 0.0, f"run failed: {error}", result
    answer = result["answer"]
    answer = answer if isinstance(answer, str) else repr(answer)
    score, feedback = score_answer(answer, example.expected, example.checker)
    return score, feedback, result


def _require_gepa():
    try:
        import rlm_gepa  # noqa: F401

        return rlm_gepa
    except ImportError as exc:  # pragma: no cover (exercised via main())
        raise ImportError(_NEEDS_GEPA) from exc


def build_project(
    dataset_path: str | Path | None = None,
    *,
    main_model: str | None = None,
    sub_model: str | None = None,
    backend: str | None = None,
):
    """Construct the RLMGepaProject that optimizes the doctrine text.

    ``dataset_path`` defaults to ``$RRLM_GEPA_DATASET``; models default to
    ``$RRLM_MAIN`` / ``$RRLM_SUB`` (Pi references, like everywhere in rrlm).
    The executor deliberately ignores the OptimizeConfig executor LMs: rrlm
    models resolve from Pi config, not litellm ids.
    """
    _require_gepa()
    from rlm_gepa import RLMGepaExampleResult, RLMGepaProject, agent_spec_from_rlm

    dataset_path = dataset_path or os.environ.get("RRLM_GEPA_DATASET")
    if not dataset_path:
        raise ValueError(
            "no dataset: pass dataset_path or set RRLM_GEPA_DATASET to a JSONL "
            "file (see rrlm.gepa module docs for the format)"
        )
    examples = load_dataset(dataset_path)
    # Domains named in RRLM_GEPA_HOLDOUT_DOMAINS never enter train or val:
    # the doctrine must not see them, or the post-hoc report (rrlm-gepa eval)
    # measures memorization instead of generalization.
    holdout_domains = os.environ.get("RRLM_GEPA_HOLDOUT_DOMAINS", "").split(",")
    in_domain, held_out = partition_domains(examples, holdout_domains)
    train, val = split_dataset(in_domain)
    if held_out:
        domains = ", ".join(sorted({e.domain for e in held_out}))
        print(
            f"holding out {len(held_out)} example(s) from domain(s) [{domains}]; "
            "score the winner on them with: rrlm-gepa eval --doctrine <winner>",
            file=sys.stderr,
        )
    main_ref = main_model or os.environ.get("RRLM_MAIN") or None
    sub_ref = sub_model or os.environ.get("RRLM_SUB") or None

    def _build_spec_rlm():
        """A throwaway harness instance so the AgentSpec derives tool and
        signature descriptions from the real thing (rlm stays source of truth)."""
        from rrlm.config import HarnessConfig, resolve_backend
        from rrlm.harness import build_lm, build_rlm
        from rrlm.pi_config import resolve_model

        main = resolve_model(main_ref)
        sub = resolve_model(sub_ref) if sub_ref else main
        cfg = HarnessConfig(
            main_model=main.ref, sub_model=sub.ref, backend=resolve_backend(backend)
        )
        main_lm = build_lm(main, min(cfg.main_max_tokens, main.max_tokens), cfg.temperature)
        sub_lm = build_lm(sub, min(cfg.sub_max_tokens, sub.max_tokens), cfg.temperature)
        return build_rlm(cfg, main_lm, sub_lm)

    class DoctrineProject(RLMGepaProject):
        project_name = "rrlm-doctrine"
        components = ("doctrine",)
        agent_spec = agent_spec_from_rlm(
            _build_spec_rlm(),
            use_cases=[
                "exact aggregation over large tabular or ledger data",
                "per-item semantic judgment at scale (classify then aggregate)",
                "exhaustive search / needle-finding across large text or code",
            ],
            runtime_grounding_examples={
                "skills": [
                    "the rlm-first-doctrine skill instructions are the optimized component",
                ],
                "sandbox facts": [
                    "inputs land as REPL variables; only previews enter model context",
                    "predict() is the typed sub-LM call; rlm_spawn spawns a child agent",
                    "independent predict() calls are batched with asyncio.gather",
                ],
                "task behaviors": [
                    "exactness tasks are solved with pure Python, zero LM calls",
                    "semantic fan-out activates only on irreducible natural text",
                    "answers are verified by an independent method before SUBMIT",
                ],
            },
            scoring_description=(
                "Score is 1.0 when the answer passes the example's machine checker "
                "(exact match, containment, or numeric tolerance), else 0.0. "
                "Feedback states the expected value, the actual answer, and for "
                "numeric checks the delta."
            ),
            counterfactual_axis_name="task shapes",
        )

        def seed_candidate(self) -> dict[str, str]:
            return {"doctrine": DOCTRINE}

        def load_trainset(self):
            return train

        def load_valset(self):
            return val

        async def evaluate_example(self, candidate, example, context):
            score, feedback, result = await score_example_with_doctrine(
                example,
                candidate["doctrine"],
                main_model=main_ref,
                sub_model=sub_ref,
                backend=backend,
                max_iterations=context.max_iterations or 30,
                timeout_s=context.task_timeout or None,
            )
            trace = result.get("trace")
            error = result["error"]
            if trace is None and not error:
                error = "no RunTrace captured for this run"
            return RLMGepaExampleResult(
                score=score,
                feedback=feedback,
                traces=[trace] if trace is not None else [],
                rlm_inputs={
                    "example_id": example.example_id,
                    "task": example.instruction,
                    "data_chars": len(example.data),
                },
                example_id=example.example_id,
                error=error if (error and trace is None) else None,
            )

    return DoctrineProject()


def eval_main(argv: list[str] | None = None) -> int:
    """``rrlm-gepa eval``: score a doctrine on held-out (or all) examples.

    The generalization report: optimize on some domains, then run this on the
    domains the optimizer never saw. Uses the exact scoring path GEPA
    optimized against (:func:`score_example_with_doctrine`), so in-domain and
    held-out numbers are comparable. Needs no rlm_gepa install - evaluation
    is just the harness plus the checkers.
    """
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        prog="rrlm-gepa eval",
        description="Score a doctrine per domain over a JSONL dataset "
                    "(defaults to the RRLM_GEPA_HOLDOUT_DOMAINS examples).",
    )
    parser.add_argument("--doctrine", default=None, metavar="PATH",
                        help="doctrine text file (e.g. a GEPA winner); default: the built-in doctrine")
    parser.add_argument("--dataset", default=os.environ.get("RRLM_GEPA_DATASET"),
                        help="JSONL dataset (env RRLM_GEPA_DATASET)")
    parser.add_argument("--domains", default=os.environ.get("RRLM_GEPA_HOLDOUT_DOMAINS", ""),
                        help="comma-separated domains to evaluate (env "
                             "RRLM_GEPA_HOLDOUT_DOMAINS); empty = every example")
    parser.add_argument("--main", dest="main_model",
                        default=os.environ.get("RRLM_MAIN") or None)
    parser.add_argument("--sub", dest="sub_model",
                        default=os.environ.get("RRLM_SUB") or None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    if not args.dataset:
        parser.error("no dataset: pass --dataset or set RRLM_GEPA_DATASET")
    doctrine_text = (
        Path(args.doctrine).read_text(encoding="utf-8") if args.doctrine else DOCTRINE
    )
    examples = load_dataset(args.dataset)
    if args.domains.strip():
        _, selected = partition_domains(examples, args.domains.split(","))
    else:
        selected = examples

    async def evaluate() -> list[dict]:
        rows = []
        for example in selected:  # sequential: comparable, budget-predictable runs
            score, feedback, result = await score_example_with_doctrine(
                example, doctrine_text,
                main_model=args.main_model, sub_model=args.sub_model,
                backend=args.backend, max_iterations=args.max_iterations,
                timeout_s=args.timeout,
            )
            rows.append({
                "example_id": example.example_id,
                "domain": example.domain or "(untagged)",
                "score": score,
                "feedback": feedback,
                "wall_clock_s": result["wall_clock_s"],
            })
        return rows

    rows = asyncio.run(evaluate())
    by_domain: dict[str, list[float]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row["score"])
    report = {
        "doctrine": args.doctrine or "(built-in)",
        "examples": len(rows),
        "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4) if rows else None,
        "domains": {
            domain: {"n": len(scores), "mean_score": round(sum(scores) / len(scores), 4)}
            for domain, scores in sorted(by_domain.items())
        },
        "rows": rows,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"doctrine: {report['doctrine']}")
        for domain, stats in report["domains"].items():
            print(f"  {domain}: {stats['mean_score']:.2f} over {stats['n']} example(s)")
        overall = report["mean_score"]
        print(f"  overall: {overall:.2f} over {report['examples']} example(s)"
              if overall is not None else "  no examples selected")
    return 0 if rows else 1


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        sys.exit(eval_main(sys.argv[2:]))
    try:
        from rlm_gepa import OptimizeConfig, run_project_cli
    except ImportError:
        print(_NEEDS_GEPA, file=sys.stderr)
        sys.exit(1)
    sys.exit(run_project_cli(build_project, OptimizeConfig()))


if __name__ == "__main__":
    main()
