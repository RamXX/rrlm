"""Task definitions. Each task carries its data, instruction, and a checker so
runs are scored mechanically and results stay comparable across models."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Task:
    task_id: str
    kind: str
    instruction: str
    data: str
    check: Callable[[str], tuple[bool, str]]  # answer -> (passed, detail)
    meta: dict = field(default_factory=dict)


def make_ledger_task(size: int = 2000, seed: int = 42) -> Task:
    """Synthetic transaction ledger; the answer requires exact aggregation.

    Plants transactions for a target user among noise lines. The correct answer
    is the sum of 'ok'-status amounts for that user -- trivially checkable, and
    big enough at scale that context-stuffing degrades while REPL code does not.
    """
    rng = random.Random(seed)
    target_user = f"u{rng.randint(100, 999)}"
    statuses = ["ok", "ok", "ok", "failed", "pending"]
    lines: list[str] = []
    expected_total = 0.0
    target_count = 0

    for i in range(size):
        if rng.random() < 0.04:
            user = target_user
        else:
            user = f"u{rng.randint(100, 999)}"
            while user == target_user:
                user = f"u{rng.randint(100, 999)}"
        amount = round(rng.uniform(1.0, 500.0), 2)
        status = rng.choice(statuses)
        if user == target_user and status == "ok":
            expected_total += amount
            target_count += 1
        ts = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
        lines.append(
            f"{ts} txn id=t{i:07d} user={user} amount={amount:.2f} status={status}"
        )

    expected_total = round(expected_total, 2)
    instruction = (
        f"The data is a transaction ledger, one transaction per line. Compute the "
        f"exact total amount of transactions with status=ok for user {target_user}. "
        f"Answer with the number rounded to 2 decimals."
    )

    def check(answer: str) -> tuple[bool, str]:
        numbers = [float(x) for x in re.findall(r"-?\d[\d,]*\.?\d*", answer.replace(",", ""))]
        for value in numbers:
            if abs(value - expected_total) < 0.01:
                return True, f"matched expected {expected_total}"
        return False, f"expected {expected_total}, answer numbers: {numbers[:5]}"

    return Task(
        task_id=f"ledger-{size}",
        kind="aggregation",
        instruction=instruction,
        data="\n".join(lines),
        check=check,
        meta={
            "n_lines": size,
            "seed": seed,
            "target_user": target_user,
            "target_txn_count": target_count,
            "expected_total": expected_total,
            "data_chars": sum(len(line) + 1 for line in lines),
        },
    )


_PEOPLE = [
    "Marta", "Tomas", "Iris", "Devon", "Priya", "Hugo", "Lena", "Sofia",
    "Andrei", "Naomi", "Kofi", "Elif", "Bruno", "Yuki", "Clara", "Omar",
]
_CITIES = [
    "Lisbon", "Oslo", "Berlin", "Kyoto", "Valparaiso", "Tallinn", "Marseille",
    "Cusco", "Galway", "Tbilisi", "Zagreb", "Maputo", "Hanoi", "Quebec",
]

_RELOCATION_TEMPLATES = [
    "{p} finally unpacked the last box in the new {c} apartment.",
    "After years in the old town, {p} now calls {c} home.",
    "{p} signed a two-year lease near the botanical garden in {c} and updated the mailing address.",
]
_NOISE_TEMPLATES = [
    "{p} dreamed of someday seeing {c} in the spring.",
    "A postcard from {c} arrived for {p} on Tuesday.",
    "{p} read a long novel set in {c}.",
    "{p} booked a weekend flight to {c} for the festival.",
    "{p} watched a documentary about the markets of {c}.",
    "{p} said the museum in {c} was overrated anyway.",
]
_FILLER = [
    "The rain did not stop until well past midnight.",
    "Somebody left the office windows open again.",
    "The bakery on the corner changed its opening hours.",
    "Nothing on the radio seemed worth listening to.",
    "The committee postponed its quarterly meeting once more.",
]


def make_needle_task(size: int = 2000, seed: int = 42) -> Task:
    """Semantic needle: exactly one sentence states (in paraphrase) where the
    target person lives now. Keyword grep is unreliable -- other people also
    relocate, and the target person appears in many non-residence contexts."""
    rng = random.Random(seed)
    target_person = rng.choice(_PEOPLE)
    target_city = rng.choice(_CITIES)
    needle = rng.choice(_RELOCATION_TEMPLATES).format(p=target_person, c=target_city)

    lines = []
    for _ in range(size - 1):
        roll = rng.random()
        if roll < 0.25:
            lines.append(rng.choice(_FILLER))
        elif roll < 0.30:
            # other people relocating to other cities (defeats one-hit grep)
            person = rng.choice([p for p in _PEOPLE if p != target_person])
            city = rng.choice([c for c in _CITIES if c != target_city])
            lines.append(rng.choice(_RELOCATION_TEMPLATES).format(p=person, c=city))
        else:
            person = rng.choice(_PEOPLE)
            city = rng.choice([c for c in _CITIES if c != target_city])
            lines.append(rng.choice(_NOISE_TEMPLATES).format(p=person, c=city))
    lines.insert(rng.randint(0, len(lines)), needle)

    def check(answer: str) -> tuple[bool, str]:
        found = target_city.lower() in answer.lower()
        short = len(answer) < 200  # reject city-name dumps
        if found and short:
            return True, f"matched {target_city}"
        return False, f"expected {target_city}, got: {answer[:120]!r}"

    return Task(
        task_id=f"needle-{size}",
        kind="semantic-retrieval",
        instruction=(
            f"The data is a collection of diary sentences, one per line. "
            f"Where does {target_person} live now? Answer with the city name only."
        ),
        data="\n".join(lines),
        check=check,
        meta={
            "size": size,
            "seed": seed,
            "target_person": target_person,
            "target_city": target_city,
            "data_chars": sum(len(line) + 1 for line in lines),
        },
    )


# Slot-filled skeletons: combinatorial space far exceeds any realistic review
# count, so deduplication cannot compress the corpus to an eyeballable set.
# Polarity is carried by the situation described, not by sentiment keywords.
_NEG_SKELETONS = [
    "Returned it {when} and asked for {refund}.",
    "It sat in the {place} after {ordinal} use, next to the {object}.",
    "My old one from {year} still does the job, so {person} got this one.",
    "Customer service heard from me {times} in the first month about the {part}.",
    "I ended up borrowing {person}'s instead for the {event}.",
    "The box now props open the {place} door, which is the most use it gets.",
    "I keep the receipt taped to the {object} as a reminder before {event}.",
    "Gave it to {person} after {ordinal} try, who handed it right back.",
]
_POS_SKELETONS = [
    "Bought a second one for {person} {when}.",
    "Since {year} it has been on my {place} shelf, used before every {event}.",
    "{person} keeps asking where I got it, most recently at the {event}.",
    "It survived {times} house moves and the {object} incident.",
    "I stopped comparing alternatives {when}, even at the {event}.",
    "The replacement I budgeted for in {year} turned out unnecessary.",
    "It is the first thing I pack for every {event}, ahead of the {object}.",
    "{person} finally admitted I was right about it {when}.",
]
_SLOTS = {
    "when": ["within the week", "the same month", "after two days", "before the trip",
             "right after the holidays", "by the second weekend"],
    "refund": ["my money back", "a refund", "store credit", "an exchange"],
    "place": ["garage", "kitchen", "hallway", "basement", "office", "pantry"],
    "ordinal": ["the second", "the third", "one single", "the first"],
    "object": ["toaster", "ladder", "blender", "printer", "bike pump", "kettle"],
    "year": ["2018", "2019", "2020", "2021", "2022"],
    "person": ["my mother", "my cousin", "my neighbor", "my partner", "a coworker",
               "my brother-in-law"],
    "times": ["twice", "three times", "four times"],
    "part": ["handle", "lid", "power cord", "base", "latch"],
    "event": ["camping trip", "family dinner", "garage sale", "office party",
              "weekend market", "move"],
}
_PRODUCTS = ["P201", "P202", "P203", "P204", "P205", "P206"]


def _render_review(rng: random.Random, skeletons: list[str]) -> str:
    skeleton = rng.choice(skeletons)
    fields = re.findall(r"{(\w+)}", skeleton)
    body = skeleton.format(**{f: rng.choice(_SLOTS[f]) for f in fields})
    # day-of-ownership prefix multiplies the combinatorial space ~360x so the
    # corpus cannot be compressed to an eyeballable unique set by dedup
    return f"Day {rng.randint(2, 364)} of owning it. {body}"


def make_reviews_task(size: int = 600, seed: int = 42) -> Task:
    """Semantic aggregation: sentiment is phrased without sentiment keywords, so
    the per-item judgment is genuinely semantic; the aggregate is mechanical."""
    rng = random.Random(seed)
    target_product = rng.choice(_PRODUCTS)
    # target gets a clearly highest negative fraction by construction
    neg_rate = {p: rng.uniform(0.10, 0.30) for p in _PRODUCTS}
    neg_rate[target_product] = 0.65

    lines, neg_counts, totals = [], dict.fromkeys(_PRODUCTS, 0), dict.fromkeys(_PRODUCTS, 0)
    for i in range(size):
        product = _PRODUCTS[i % len(_PRODUCTS)]
        totals[product] += 1
        if rng.random() < neg_rate[product]:
            text = _render_review(rng, _NEG_SKELETONS)
            neg_counts[product] += 1
        else:
            text = _render_review(rng, _POS_SKELETONS)
        lines.append(f'product={product} review_id=r{i:05d} text="{text}"')
    rng.shuffle(lines)

    fractions = {p: neg_counts[p] / totals[p] for p in _PRODUCTS}
    actual_max = max(fractions, key=fractions.get)  # construction makes this the target

    def check(answer: str) -> tuple[bool, str]:
        mentioned = [p for p in _PRODUCTS if p in answer]
        if mentioned == [actual_max]:
            return True, f"matched {actual_max} (neg fraction {fractions[actual_max]:.2f})"
        return False, f"expected {actual_max}, answer mentioned {mentioned}"

    return Task(
        task_id=f"reviews-{size}",
        kind="semantic-aggregation",
        instruction=(
            "The data is a list of product reviews, one per line. Which product id "
            "has the highest fraction of negative reviews? Answer with the product id only."
        ),
        data="\n".join(lines),
        check=check,
        meta={
            "size": size,
            "seed": seed,
            "target_product": actual_max,
            "neg_fractions": {p: round(f, 3) for p, f in fractions.items()},
            "data_chars": sum(len(line) + 1 for line in lines),
        },
    )


# (correct_body, buggy_body) pairs; bodies are formatted with the function name
_FUNCTION_BANK: list[tuple[str, str, str]] = [
    (
        "sum_to",
        "def {name}(n):\n    return sum(range(1, n + 1))\n",
        "def {name}(n):\n    return sum(range(1, n))\n",
    ),
    (
        "factorial",
        "def {name}(n):\n    out = 1\n    for k in range(2, n + 1):\n        out *= k\n    return out\n",
        "def {name}(n):\n    out = 1\n    for k in range(2, n):\n        out *= k\n    return out\n",
    ),
    (
        "count_vowels",
        "def {name}(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
        "def {name}(s):\n    return sum(1 for ch in s.lower() if ch in 'aeio')\n",
    ),
    (
        "is_leap",
        "def {name}(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n",
        "def {name}(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 200 == 0)\n",
    ),
    (
        "median",
        "def {name}(xs):\n    s = sorted(xs)\n    m = len(s) // 2\n    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n",
        "def {name}(xs):\n    s = sorted(xs)\n    m = len(s) // 2\n    return s[m] if len(s) % 2 else (s[m] + s[m + 1]) / 2\n",
    ),
    (
        "clamp",
        "def {name}(x, lo, hi):\n    return lo if x < lo else hi if x > hi else x\n",
        "def {name}(x, lo, hi):\n    return hi if x < lo else hi if x > hi else x\n",
    ),
    (
        "reverse_words",
        "def {name}(s):\n    return ' '.join(s.split()[::-1])\n",
        "def {name}(s):\n    return ' '.join(w[::-1] for w in s.split()[::-1])\n",
    ),
    (
        "mean",
        "def {name}(xs):\n    return sum(xs) / len(xs)\n",
        "def {name}(xs):\n    return sum(xs) / (len(xs) - 1)\n",
    ),
    (
        "max_run",
        "def {name}(xs):\n    best = cur = 1\n    for a, b in zip(xs, xs[1:]):\n        cur = cur + 1 if a == b else 1\n        best = max(best, cur)\n    return best\n",
        "def {name}(xs):\n    best = cur = 1\n    for a, b in zip(xs, xs[1:]):\n        cur = cur + 1 if a == b else 0\n        best = max(best, cur)\n    return best\n",
    ),
    (
        "dot",
        "def {name}(a, b):\n    return sum(x * y for x, y in zip(a, b))\n",
        "def {name}(a, b):\n    return sum(x * y for x, y in zip(a, b[1:]))\n",
    ),
]


def make_bugfind_task(size: int = 60, seed: int = 42) -> Task:
    """Code reasoning: a module of small functions, exactly one buggy. The REPL
    condition can import and property-test them; the baseline must read."""
    rng = random.Random(seed)
    buggy_index = rng.randint(0, size - 1)
    chunks, buggy_name = [], None
    for i in range(size):
        base, correct, buggy = _FUNCTION_BANK[i % len(_FUNCTION_BANK)]
        name = f"fn_{i:03d}_{base}"
        if i == buggy_index:
            chunks.append(buggy.format(name=name))
            buggy_name = name
        else:
            chunks.append(correct.format(name=name))
    rng.shuffle(chunks)

    def check(answer: str) -> tuple[bool, str]:
        named = set(re.findall(r"fn_\d+_\w+", answer))
        if named == {buggy_name}:
            return True, f"matched {buggy_name}"
        return False, f"expected {buggy_name}, answer named {sorted(named)}"

    return Task(
        task_id=f"bugfind-{size}",
        kind="code-reasoning",
        instruction=(
            "The data is the source of a Python module. Exactly one function has a "
            "bug (it returns a wrong result for some valid input). Name the buggy "
            "function. Answer with the function name only."
        ),
        data="\n".join(chunks),
        check=check,
        meta={
            "size": size,
            "seed": seed,
            "buggy_function": buggy_name,
            "data_chars": sum(len(c) + 1 for c in chunks),
        },
    )


def make_imdb_task(size: int = 1000, seed: int = 42, cache_path=None) -> Task:
    """Real-text semantic aggregation: genuine IMDB reviews (true labels as
    ground truth), product ids assigned with a planted negative-rate skew.
    Human text resists template induction, so per-item judgment is semantic."""
    from rrlm.corpora import IMDB_CACHE, fetch_imdb_pool

    pool = fetch_imdb_pool(cache_path or IMDB_CACHE)
    rng = random.Random(seed)
    neg_pool = pool["neg"][:]
    pos_pool = pool["pos"][:]
    rng.shuffle(neg_pool)
    rng.shuffle(pos_pool)

    target_product = rng.choice(_PRODUCTS)
    neg_rate = {p: rng.uniform(0.10, 0.30) for p in _PRODUCTS}
    neg_rate[target_product] = 0.65

    lines, neg_counts, totals = [], dict.fromkeys(_PRODUCTS, 0), dict.fromkeys(_PRODUCTS, 0)
    for i in range(size):
        product = _PRODUCTS[i % len(_PRODUCTS)]
        totals[product] += 1
        if rng.random() < neg_rate[product]:
            text = neg_pool.pop()
            neg_counts[product] += 1
        else:
            text = pos_pool.pop()
        text = text.replace('"', "'")
        lines.append(f'product={product} review_id=r{i:05d} text="{text}"')
    rng.shuffle(lines)

    fractions = {p: neg_counts[p] / totals[p] for p in _PRODUCTS}
    actual_max = max(fractions, key=fractions.get)

    def check(answer: str) -> tuple[bool, str]:
        mentioned = [p for p in _PRODUCTS if p in answer]
        if mentioned == [actual_max]:
            return True, f"matched {actual_max} (neg fraction {fractions[actual_max]:.2f})"
        return False, f"expected {actual_max}, answer mentioned {mentioned}"

    return Task(
        task_id=f"imdb-{size}",
        kind="semantic-aggregation-natural",
        instruction=(
            "The data is a list of product reviews, one per line. Which product id "
            "has the highest fraction of negative reviews? Answer with the product id only."
        ),
        data="\n".join(lines),
        check=check,
        meta={
            "size": size,
            "seed": seed,
            "target_product": actual_max,
            "neg_fractions": {p: round(f, 3) for p, f in fractions.items()},
            "data_chars": sum(len(line) + 1 for line in lines),
        },
    )


TASK_BUILDERS: dict[str, Callable[..., Task]] = {
    "ledger": make_ledger_task,
    "needle": make_needle_task,
    "reviews": make_reviews_task,
    "bugfind": make_bugfind_task,
    "imdb": make_imdb_task,
}
