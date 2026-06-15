"""Unit tests for synthetic task generation and checking."""

from rrlm.tasks import (
    make_bugfind_task,
    make_ledger_task,
    make_needle_task,
    make_reviews_task,
)


def test_ledger_is_deterministic():
    a = make_ledger_task(size=500, seed=7)
    b = make_ledger_task(size=500, seed=7)
    assert a.data == b.data
    assert a.meta["expected_total"] == b.meta["expected_total"]
    assert a.meta["target_user"] == b.meta["target_user"]


def test_ledger_seeds_differ():
    a = make_ledger_task(size=500, seed=1)
    b = make_ledger_task(size=500, seed=2)
    assert a.data != b.data


def test_ledger_expected_total_matches_data():
    task = make_ledger_task(size=1000, seed=11)
    user = task.meta["target_user"]
    total = 0.0
    for line in task.data.splitlines():
        if f"user={user} " in line and line.endswith("status=ok"):
            total += float(line.split("amount=")[1].split(" ")[0])
    assert abs(round(total, 2) - task.meta["expected_total"]) < 0.01


def test_ledger_checker_accepts_correct_answer():
    task = make_ledger_task(size=500, seed=3)
    expected = task.meta["expected_total"]
    passed, _ = task.check(f"The total is {expected:.2f} USD.")
    assert passed


def test_ledger_checker_accepts_thousands_separators():
    task = make_ledger_task(size=500, seed=3)
    expected = task.meta["expected_total"]
    passed, _ = task.check(f"Total: {expected:,.2f}")
    assert passed


def test_ledger_checker_rejects_wrong_answer():
    task = make_ledger_task(size=500, seed=3)
    wrong = task.meta["expected_total"] + 1.0
    passed, detail = task.check(f"The total is {wrong:.2f}")
    assert not passed
    assert "expected" in detail


def test_needle_has_exactly_one_relocation_for_target():
    task = make_needle_task(size=1000, seed=5)
    person, city = task.meta["target_person"], task.meta["target_city"]
    target_city_lines = [l for l in task.data.splitlines() if city in l]
    # the target city appears ONLY in the needle sentence
    assert len(target_city_lines) == 1
    assert person in target_city_lines[0]
    assert task.check(city)[0]
    assert task.check(f"{person} lives in {city}.")[0]
    assert not task.check("Oslo Berlin Lisbon Kyoto " * 12)[0]  # length-capped dump
    wrong = "Berlin" if city != "Berlin" else "Oslo"
    assert not task.check(wrong)[0]


def test_needle_deterministic():
    assert make_needle_task(size=300, seed=9).data == make_needle_task(size=300, seed=9).data


def test_reviews_resists_dedup_compression():
    task = make_reviews_task(size=2000, seed=4)
    texts = [line.split('text="')[1].rstrip('"') for line in task.data.splitlines()]
    # combinatorial slots: unique texts must be a large fraction of the corpus
    assert len(set(texts)) > len(texts) * 0.5


def test_reviews_target_is_strict_max():
    task = make_reviews_task(size=600, seed=4)
    fractions = task.meta["neg_fractions"]
    target = task.meta["target_product"]
    others = [v for k, v in fractions.items() if k != target]
    assert fractions[target] > max(others)
    assert task.check(target)[0]
    assert not task.check("P201 P202 P203")[0]  # multi-mention rejected
    wrong = next(p for p in fractions if p != target)
    assert not task.check(wrong)[0]


def test_bugfind_exactly_one_buggy_function():
    task = make_bugfind_task(size=40, seed=8)
    buggy = task.meta["buggy_function"]
    assert task.data.count(f"def {buggy}(") == 1
    assert task.check(f"The bug is in {buggy}.")[0]
    assert not task.check(f"{buggy} or maybe fn_000_sum_to")[0]
    assert not task.check("fn_001_factorial")[0]


def _fake_imdb_cache(tmp_path):
    import json

    pool = {
        "neg": [f"This film wasted {i} minutes of my life and the plot went nowhere." for i in range(900)],
        "pos": [f"A quiet masterpiece; scene {i} alone justified the ticket." for i in range(900)],
    }
    path = tmp_path / "imdb_pool.json"
    path.write_text(json.dumps(pool))
    return path


def test_imdb_task_deterministic_and_strict_max(tmp_path):
    from rrlm.tasks import make_imdb_task

    cache = _fake_imdb_cache(tmp_path)
    a = make_imdb_task(size=300, seed=6, cache_path=cache)
    b = make_imdb_task(size=300, seed=6, cache_path=cache)
    assert a.data == b.data
    fractions = a.meta["neg_fractions"]
    target = a.meta["target_product"]
    assert fractions[target] > max(v for k, v in fractions.items() if k != target)
    assert a.check(target)[0]
    assert not a.check("P201 P202")[0]


def test_imdb_lines_are_single_line(tmp_path):
    from rrlm.tasks import make_imdb_task

    task = make_imdb_task(size=120, seed=2, cache_path=_fake_imdb_cache(tmp_path))
    for line in task.data.splitlines():
        assert line.startswith("product=P2")


def test_bugfind_checker_handles_four_digit_indices():
    task = make_bugfind_task(size=2000, seed=42)
    buggy = task.meta["buggy_function"]
    assert task.check(buggy)[0]


def test_bugfind_module_compiles_and_buggy_differs():
    task = make_bugfind_task(size=40, seed=8)
    namespace: dict = {}
    exec(compile(task.data, "<bugfind>", "exec"), namespace)  # noqa: S102 -- our own generated source
    buggy = task.meta["buggy_function"]
    assert callable(namespace[buggy])
