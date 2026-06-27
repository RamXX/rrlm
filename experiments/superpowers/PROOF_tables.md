# RLM-superpower matrix -- results

## bugfind -- pitune

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 60 | 42 | baseline | True | completed | 21.81 | 2372 | matched fn_040_sum_to |
| 60 | 42 | rlm | True | completed | 718.66 | 53035 | matched fn_040_sum_to |

## imdb -- pitune

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 200 | 42 | baseline | True | completed | 249.13 | 25584 | matched P205 (neg fraction 0.55) |
| 200 | 42 | rlm | True | completed | 363.79 | 86928 | matched P205 (neg fraction 0.55) |
| 200 | 42 | rlm-self | True | completed | 1111.22 | 102059 | matched P205 (neg fraction 0.55) |
| 1500 | 42 | baseline | False | error | 0.66 | 0 | no answer |
| 1500 | 42 | rlm | True | completed | 1402.91 | 69271 | matched P205 (neg fraction 0.67) |

## ledger -- pitune

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 100 | 42 | baseline | True | completed | 35.49 | 4794 | matched expected 584.04 |
| 100 | 42 | rlm | True | completed | 73.15 | 11521 | matched expected 584.04 |
| 100 | 43 | baseline | False | completed | 40.52 | 4799 | expected 430.33, answer numbers: [420.33] |
| 100 | 43 | rlm | True | completed | 107.16 | 17851 | matched expected 430.33 |
| 100 | 44 | baseline | False | completed | 43.69 | 4789 | expected 1527.29, answer numbers: [1722.39] |
| 100 | 44 | rlm | True | completed | 70.19 | 11389 | matched expected 1527.29 |
| 300 | 42 | baseline | False | completed | 126.96 | 13947 | expected 2039.16, answer numbers: [2594.07] |
| 300 | 42 | rlm | True | completed | 160.19 | 18135 | matched expected 2039.16 |
| 300 | 43 | baseline | False | completed | 113.9 | 13967 | expected 2235.95, answer numbers: [2375.16] |
| 300 | 43 | rlm | True | completed | 100.33 | 11489 | matched expected 2235.95 |
| 300 | 44 | baseline | False | completed | 124.6 | 13953 | expected 1977.35, answer numbers: [2807.19] |
| 300 | 44 | rlm | True | completed | 113.41 | 11968 | matched expected 1977.35 |
| 500 | 42 | baseline | False | completed | 236.67 | 23082 | expected 2743.79, answer numbers: [3501.97] |
| 500 | 42 | rlm | True | completed | 158.7 | 18542 | matched expected 2743.79 |
| 600 | 42 | baseline | False | completed | 135.99 | 27659 | expected 2996.6, answer numbers: [3597.06] |
| 600 | 42 | rlm | True | completed | 97.41 | 12305 | matched expected 2996.6 |
| 600 | 43 | baseline | False | completed | 140.28 | 27698 | expected 4714.03, answer numbers: [4256.78] |
| 600 | 43 | rlm | True | completed | 107.04 | 17777 | matched expected 4714.03 |
| 600 | 44 | baseline | False | completed | 147.81 | 27683 | expected 2929.44, answer numbers: [3601.52] |
| 600 | 44 | rlm | True | completed | 129.07 | 18245 | matched expected 2929.44 |
| 1000 | 42 | baseline | False | completed | 230.86 | 45964 | expected 4686.21, answer numbers: [5698.72] |
| 1000 | 42 | rlm | True | completed | 91.7 | 11530 | matched expected 4686.21 |
| 1000 | 43 | baseline | False | completed | 266.17 | 46026 | expected 6739.76, answer numbers: [5687.92] |
| 1000 | 43 | rlm | True | completed | 89.05 | 11447 | matched expected 6739.76 |
| 1000 | 44 | baseline | False | completed | 257.12 | 45997 | expected 5750.58, answer numbers: [5328.04] |
| 1000 | 44 | rlm | True | completed | 112.19 | 11785 | matched expected 5750.58 |
| 1300 | 42 | baseline | False | completed | 209.42 | 59704 | expected 6529.69, answer numbers: [5819.2] |
| 1300 | 42 | rlm | True | completed | 140.74 | 11631 | matched expected 6529.69 |
| 1300 | 43 | baseline | False | completed | 244.42 | 59753 | expected 9246.06, answer numbers: [7028.31] |
| 1300 | 43 | rlm | True | completed | 130.08 | 11503 | matched expected 9246.06 |
| 1300 | 44 | baseline | False | completed | 220.76 | 59744 | expected 8690.14, answer numbers: [8065.72] |
| 1300 | 44 | rlm | True | completed | 163.83 | 12170 | matched expected 8690.14 |
| 2000 | 42 | baseline | False | error | 0.67 | 0 | no answer |
| 2000 | 42 | rlm | True | completed | 207.54 | 19153 | matched expected 10143.63 |
| 2000 | 43 | baseline | False | error | 2.97 | 0 | no answer |
| 2000 | 43 | rlm | True | completed | 114.0 | 11795 | matched expected 14480.11 |
| 5000 | 42 | baseline | False | error | 0.65 | 0 | no answer |
| 5000 | 42 | rlm | True | completed | 105.33 | 11640 | matched expected 26453.9 |
| 5000 | 43 | baseline | False | error | 0.53 | 0 | no answer |
| 5000 | 43 | rlm | True | completed | 82.21 | 11564 | matched expected 33663.18 |
| 20000 | 42 | baseline | False | error | 0.86 | 0 | no answer |
| 20000 | 42 | rlm | True | completed | 199.7 | 25186 | matched expected 115797.87 |

## ledger -- qwen-official

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 2000 | 42 | baseline | False | completed | 1237.52 | 91765 | expected 10143.63, answer numbers: [10666.61] |
| 2000 | 42 | rlm | True | completed | 113.65 | 11720 | matched expected 10143.63 |
| 20000 | 42 | baseline | False | error | 1853.44 | 0 | no answer |
| 20000 | 42 | rlm | True | completed | 206.14 | 11585 | matched expected 115797.87 |

## needle -- pitune

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 2000 | 42 | baseline | True | completed | 329.59 | 25514 | matched Lisbon |
| 2000 | 42 | rlm | True | completed | 207.42 | 19362 | matched Lisbon |

## Evidence: what each condition actually did

### RLM solved `ledger` (size 500) by writing code:

```python
lines = data.strip().splitlines()
print(len(lines))
print(lines[:3])
print(lines[-3:])

import re
total = 0.0
count = 0
for line in lines:
    m_user = re.search(r'\buser=(\S+)', line)
    m_status = re.search(r'\bstatus=(\S+)', line)
    m_amount = re.search(r'\bamount=(\S+)', line)
    if m_user and m_status and m_amount:
        user = m_user.group(1)
        status = m_status.group(1)
        amount = float(m_amount.group(1))
        if user == 'u754' and status == 'ok':
            total += amount
            count += 1

answer = round(total, 2)
print(count, answer)

# --- next REPL turn ---
import re

lines = data.strip().splitlines()
matches = []
for line in lines:
    m_user = re.search(r'\buser=(\S+)', line)
    m_status = re.search(r'\bstatus=(\S+)', line)
    m_amount = re.search(r'\bamount=(\S+)', line)
    m_id = re.search(r'\bid=(\S+)', line)
    if m_user and m_status and m_amount and m_id:
        user = m_user.group(1)
        status = m_status.group(1)
        amount = float(m_amount.group(1))
        txn_id = m_id.group(1)
        if user == 'u754' and status == 'ok':
            matches.append((txn_id, amount))

total = sum(amount for _, amount in matches)
answer = round(total, 2)
print(len(matches), total, answer)
for txn_id, amount in matches:
    print(txn_id, amount)

# --- next REPL turn ---
SUBMIT(answer=2743.79)
```

### Baseline on `ledger` (size 2000) failed:

`BadRequestError: litellm.BadRequestError: OpenAIException - request (91764 tokens) exceeds the available context size (65536 tokens), try increasing it`

### RLM solved `needle` (size 2000) by writing code:

```python
lines = data.splitlines()
print(len(lines))
devon_lines = [line for line in lines if 'Devon' in line]
print(len(devon_lines))
for line in devon_lines:
    print(line)

# --- next REPL turn ---
lines = data.splitlines()
devon_indices = [i for i, line in enumerate(lines) if 'Devon' in line]
print(f"Total Devon mentions: {len(devon_indices)}")
print(f"Last Devon mention index: {devon_indices[-1]}")
print(f"Last Devon mention: {lines[devon_indices[-1]]}")

lease_line = "Devon signed a two-year lease near the botanical garden in Lisbon and updated the mailing address."
try:
    lease_index = lines.index(lease_line)
    print(f"Lisbon lease index: {lease_index}")
    later_devon_lines = [lines[i] for i in devon_indices if i > lease_index]
    print(f"Devon mentions after Lisbon lease: {len(later_devon_lines)}")
    for line in later_devon_lines:
        print(line)
except ValueError:
    print("Lisbon lease line not found exactly as expected.")

# --- next REPL turn ---
lines = data.splitlines()
lisbon_lines = [(i, line) for i, line in enumerate(lines) if 'Lisbon' in line]
print(f"Total Lisbon mentions: {len(lisbon_lines)}")
for idx, line in lisbon_lines:
    print(idx, line)

SUBMIT(answer="Lisbon")
```

### RLM solved `bugfind` (size 60) by writing code:

```python
print(len(data))
print(data.count('def fn_'))
functions = []
for line in data.splitlines():
    if line.startswith('def fn_'):
        functions.append(line.split('(')[0].replace('def ', ''))
print(functions)

# --- next REPL turn ---
import re
from math import inf

# Parse function definitions
func_defs = []
current_name = None
current_body = []
for line in data.splitlines():
    if line.startswith('def fn_'):
        if current_name:
            func_defs.append((current_name, '\n'.join(current_body)))
        current_name = line.split('(')[0].replace('def ', '')
        current_body = [line]
    elif current_name:
        current_body.append(line)
if current_name:
    func_defs.append((current_name, '\n'.join(current_body)))

print(f"Parsed {len(func_defs)} functions")

# Define correct reference implementations for testing
def ref_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def ref_is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

def ref_clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ref_factorial(n):
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out

def ref_reverse_words(s):
    return ' '.join(s.split()[::-1])

def ref_sum_to(n):
    return sum(range(1, n + 1))

def ref_median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

def ref_mean(xs):
    return sum(xs) / len(xs)

def ref_max_run(xs):
    best = 0
    cur = 0
    for x in xs:
        if x > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur
```

### RLM solved `imdb` (size 200) by writing code:

```python
import re
from collections import defaultdict

reviews = data.strip().split('\n')
print(f"Total reviews: {len(reviews)}")
print("First 3 reviews:")
for r in reviews[:3]:
    print(r[:200])

product_ids = set()
parsed_reviews = []
for line in reviews:
    match = re.match(r'product=(\w+) review_id=\w+ text="(.*)"', line)
    if match:
        pid, text = match.groups()
        product_ids.add(pid)
        parsed_reviews.append((pid, text))

print(f"Unique products: {len(product_ids)}")
print(sorted(product_ids)[:10])

# --- next REPL turn ---
import asyncio
from collections import defaultdict

# Group reviews by product ID
reviews_by_product = defaultdict(list)
for pid, text in parsed_reviews:
    reviews_by_product[pid].append(text)

print("Reviews per product:")
for pid in sorted(reviews_by_product):
    print(f"{pid}: {len(reviews_by_product[pid])}")

# Classify a sample review to test predict
sample_pid = 'P201'
sample_text = reviews_by_product[sample_pid][0]
result = await predict(
    "text: str -> is_negative: bool",
    instructions="Classify the review as negative (True) or not negative (False). Negative means the reviewer expresses dissatisfaction, criticism, or a bad experience.",
    text=sample_text
)
print(f"Sample classification for {sample_pid}: {result.is_negative}")

# --- next REPL turn ---
import asyncio
from collections import defaultdict

# Classify all reviews
tasks = [
    predict(
        "text: str -> is_negative: bool",
        instructions="Classify the review as negative (True) or not negative (False). Negative means the reviewer expresses dissat
```

### Baseline on `imdb` (size 1500) failed:

`BadRequestError: litellm.BadRequestError: OpenAIException - request (186981 tokens) exceeds the available context size (65536 tokens), try increasing it`

