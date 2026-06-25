"""Real-use-case eval: exact aggregation over a tabular dataset.

Generates a realistic orders CSV (seeded, ~6000 rows), then asks the harness for
an exact revenue total over a filtered subset. The ground truth is recomputed in
pure Python, so this verifies the path synthetic semantic data never forced: being
exact over many rows, where free-form reading silently miscounts but REPL code
does not. The CSV is too large to want in a chat context -- the everyday "too big
for context" case.

    python examples/eval_tabular.py            # uses your Pi default model
    RRLM_MAIN=openrouter/qwen/qwen3.6-27b python examples/eval_tabular.py
"""

from __future__ import annotations

import random
import re
import sys

REGIONS = ["EMEA", "AMER", "APAC", "LATAM"]
STATUSES = ["completed", "pending", "cancelled", "refunded"]
PRODUCTS = ["widget", "gadget", "sprocket", "cog", "flange", "gasket", "valve", "rotor"]

TARGET_REGION = "EMEA"
TARGET_STATUS = "completed"


def build(size: int = 6000, seed: int = 42) -> tuple[str, str, float]:
    rng = random.Random(seed)
    rows = ["order_id,order_date,region,product,qty,unit_price,status"]
    truth = 0.0
    for i in range(size):
        region = rng.choice(REGIONS)
        status = rng.choice(STATUSES)
        qty = rng.randint(1, 20)
        unit_price = round(rng.uniform(2.0, 499.0), 2)
        day = rng.randint(1, 28)
        month = rng.randint(1, 12)
        rows.append(
            f"ORD-{100000 + i},2026-{month:02d}-{day:02d},{region},"
            f"{rng.choice(PRODUCTS)},{qty},{unit_price:.2f},{status}"
        )
        if region == TARGET_REGION and status == TARGET_STATUS:
            truth += qty * unit_price
    instruction = (
        f"This is a CSV of orders. Compute the total revenue (qty * unit_price) "
        f"for orders whose region is '{TARGET_REGION}' AND status is "
        f"'{TARGET_STATUS}'. Round to 2 decimals. Answer with just the number "
        f"(no currency symbol, no commas)."
    )
    return instruction, "\n".join(rows), round(truth, 2)


def main() -> int:
    from _harness import run_eval

    instruction, data, expected = build()

    def check(answer: str) -> tuple[bool, str]:
        m = re.search(r"-?\d[\d,]*\.?\d*", answer.replace(",", ""))
        if not m:
            return False, f"no number in answer; expected {expected}"
        got = float(m.group())
        ok = abs(got - expected) < 0.02
        return ok, f"got {got}, expected {expected}"

    print(f"[tabular] ground-truth revenue for {TARGET_STATUS}/{TARGET_REGION}: {expected}")
    return 0 if run_eval("tabular", instruction, data, check) else 1


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    sys.exit(main())
