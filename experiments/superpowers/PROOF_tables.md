# RLM-superpower matrix, results

## bugfind, ornith

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 60 | 42 | baseline | True | completed | 4.76 | 2372 | matched fn_040_sum_to |
| 60 | 42 | rlm | True | completed | 54.93 | 33967 | matched fn_040_sum_to |

## imdb, ornith

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 200 | 42 | baseline | True | completed | 58.31 | 25584 | matched P205 (neg fraction 0.55) |
| 200 | 42 | rlm | True | completed | 158.29 | 81851 | matched P205 (neg fraction 0.55) |
| 200 | 42 | rlm-self | True | completed | 177.92 | 78164 | matched P205 (neg fraction 0.55) |
| 1500 | 42 | baseline | False | error | 0.63 | 0 | no answer |
| 1500 | 42 | rlm | True | completed | 757.39 | 331912 | matched P205 (neg fraction 0.67) |

## ledger, ornith

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 100 | 42 | baseline | True | completed | 6.95 | 4794 | matched expected 584.04 |
| 100 | 42 | rlm | True | completed | 15.33 | 11542 | matched expected 584.04 |
| 100 | 43 | baseline | False | completed | 7.62 | 4799 | expected 430.33, answer numbers: [441.05] |
| 100 | 43 | rlm | True | completed | 19.63 | 11647 | matched expected 430.33 |
| 100 | 44 | baseline | False | completed | 7.34 | 4789 | expected 1527.29, answer numbers: [1212.29] |
| 100 | 44 | rlm | True | completed | 15.09 | 11434 | matched expected 1527.29 |
| 300 | 42 | baseline | False | completed | 20.82 | 13947 | expected 2039.16, answer numbers: [2173.55] |
| 300 | 42 | rlm | True | completed | 14.66 | 11673 | matched expected 2039.16 |
| 300 | 43 | baseline | True | completed | 50.36 | 13967 | matched expected 2235.95 |
| 300 | 43 | rlm | True | completed | 16.45 | 11475 | matched expected 2235.95 |
| 300 | 44 | baseline | False | completed | 22.03 | 13953 | expected 1977.35, answer numbers: [2084.76] |
| 300 | 44 | rlm | True | completed | 24.37 | 17700 | matched expected 1977.35 |
| 500 | 42 | baseline | False | completed | 33.31 | 23082 | expected 2743.79, answer numbers: [2814.48] |
| 500 | 42 | rlm | True | completed | 18.13 | 11482 | matched expected 2743.79 |
| 600 | 42 | baseline | False | completed | 29.58 | 27659 | expected 2996.6, answer numbers: [2754.83] |
| 600 | 42 | rlm | True | completed | 15.55 | 11392 | matched expected 2996.6 |
| 600 | 43 | baseline | False | completed | 31.71 | 27698 | expected 4714.03, answer numbers: [3010.35] |
| 600 | 43 | rlm | True | completed | 16.46 | 11519 | matched expected 4714.03 |
| 600 | 44 | baseline | False | completed | 31.52 | 27683 | expected 2929.44, answer numbers: [2467.46] |
| 600 | 44 | rlm | True | completed | 18.02 | 11663 | matched expected 2929.44 |
| 1000 | 42 | baseline | False | completed | 54.28 | 45964 | expected 4686.21, answer numbers: [4537.98] |
| 1000 | 42 | rlm | True | completed | 15.71 | 11507 | matched expected 4686.21 |
| 1000 | 43 | baseline | False | completed | 56.32 | 46026 | expected 6739.76, answer numbers: [3836.57] |
| 1000 | 43 | rlm | True | completed | 28.81 | 24110 | matched expected 6739.76 |
| 1000 | 44 | baseline | False | completed | 56.22 | 45997 | expected 5750.58, answer numbers: [3635.92] |
| 1000 | 44 | rlm | True | completed | 32.05 | 24940 | matched expected 5750.58 |
| 1300 | 42 | baseline | False | completed | 53.95 | 59704 | expected 6529.69, answer numbers: [5547.96] |
| 1300 | 42 | rlm | True | completed | 23.25 | 17736 | matched expected 6529.69 |
| 1300 | 43 | baseline | False | completed | 53.53 | 59753 | expected 9246.06, answer numbers: [5855.61] |
| 1300 | 43 | rlm | True | completed | 33.42 | 24804 | matched expected 9246.06 |
| 1300 | 44 | baseline | False | completed | 55.46 | 59744 | expected 8690.14, answer numbers: [5739.33] |
| 1300 | 44 | rlm | True | completed | 16.18 | 11465 | matched expected 8690.14 |
| 2000 | 42 | baseline | False | error | 0.35 | 0 | no answer |
| 2000 | 42 | rlm | True | completed | 13.06 | 11415 | matched expected 10143.63 |
| 2000 | 43 | baseline | False | error | 0.46 | 0 | no answer |
| 2000 | 43 | rlm | True | completed | 15.41 | 11481 | matched expected 14480.11 |
| 5000 | 42 | baseline | False | error | 0.39 | 0 | no answer |
| 5000 | 42 | rlm | True | completed | 32.49 | 31146 | matched expected 26453.9 |
| 5000 | 43 | baseline | False | error | 0.46 | 0 | no answer |
| 5000 | 43 | rlm | True | completed | 18.34 | 17302 | matched expected 33663.18 |
| 20000 | 42 | baseline | False | error | 0.75 | 0 | no answer |
| 20000 | 42 | rlm | True | completed | 43.72 | 31457 | matched expected 115797.87 |

## needle, ornith

| size | seed | condition | passed | status | wall_s | prompt_tok | detail |
|---|---|---|---|---|---|---|---|
| 2000 | 42 | baseline | True | completed | 46.09 | 25514 | matched Lisbon |
| 2000 | 42 | rlm | True | completed | 21.12 | 18409 | matched Lisbon |

## Evidence: what each condition actually did

### RLM solved `ledger` (size 500) by writing code:

```python
import re

# Parse all transactions
total = 0.0
count = 0
for line in data.strip().split('\n'):
    line = line.strip()
    if not line:
        continue
    # Extract user, amount, status
    user_match = re.search(r'user=(\S+)', line)
    amount_match = re.search(r'amount=(\S+)', line)
    status_match = re.search(r'status=(\S+)', line)
    
    if user_match and amount_match and status_match:
        user = user_match.group(1)
        amount = float(amount_match.group(1))
        status = status_match.group(1)
        
        if user == 'u754' and status == 'ok':
            total += amount
            count += 1

print(f"Found {count} transactions for u754 with status=ok")
print(f"Total: {total}")
print(f"Rounded: {round(total, 2)}")

# --- next REPL turn ---
SUBMIT(answer=2743.79)
```

### Baseline on `ledger` (size 2000) failed:

`BadRequestError: litellm.BadRequestError: OpenAIException - request (91764 tokens) exceeds the available context size (65536 tokens), try increasing it`

### RLM solved `needle` (size 2000) by writing code:

```python
# First, let's find all lines mentioning Devon
lines = data.strip().split('\n')
print(f"Total lines: {len(lines)}")

devon_lines = [line for line in lines if 'Devon' in line]
print(f"Lines mentioning Devon: {len(devon_lines)}")

for line in devon_lines:
    print(line)

# --- next REPL turn ---
# Find the specific sentence about Devon signing a lease
devon_lease = [line for line in devon_lines if 'signed a two-year lease' in line]
print(f"Lease sentences for Devon: {len(devon_lease)}")
for line in devon_lease:
    print(line)

# --- next REPL turn ---
SUBMIT(answer="Lisbon")
```

### RLM solved `bugfind` (size 60) by writing code:

```python
# First, let me extract all function definitions and their bodies
import re

# Find all function definitions
func_pattern = r'def (\w+)\(([^)]*)\):\s*\n((?:        .+\n)+)'
matches = re.findall(func_pattern, data)

print(f"Found {len(matches)} functions")
for name, params, body in matches[:5]:
    print(f"\n{name}({params}):")
    print(body[:200])

# --- next REPL turn ---
# Let me see the full data
print(data[:3000])

# --- next REPL turn ---
# Let me see the rest of the data
print(data[3000:])

# --- next REPL turn ---
# Let me verify fn_040_sum_to is buggy by comparing it with a correct version
# First, let me extract and execute fn_040_sum_to

# Execute all functions from the data
exec(data)

# Test fn_040_sum_to
print(f"fn_040_sum_to(5) = {fn_040_sum_to(5)}")  # Should be 15 (1+2+3+4+5)
print(f"fn_000_sum_to(5) = {fn_000_sum_to(5)}")  # Should be 15

# The buggy one returns sum(range(1, n)) = 1+2+3+4 = 10 instead of 15
print(f"\nExpected: 15, Got: {fn_040_sum_to(5)}")
print(f"fn_040_sum_to has bug: {fn_040_sum_to(5) != 15}")

# --- next REPL turn ---
SUBMIT(answer="fn_040_sum_to")
```

### RLM solved `imdb` (size 200) by writing code:

```python
# First, let's explore the data structure
lines = data.strip().split('\n')
print(f"Total lines: {len(lines)}")
print(f"\nFirst 3 lines:")
for line in lines[:3]:
    print(line[:300])
    print("---")

# --- next REPL turn ---
import re
from collections import defaultdict

# Parse all reviews
reviews = []
for line in lines:
    # Extract product ID
    product_match = re.search(r'product=(P\d+)', line)
    text_match = re.search(r'text="(.*)"', line, re.DOTALL)
    
    if product_match and text_match:
        product_id = product_match.group(1)
        review_text = text_match.group(1)
        reviews.append({'product': product_id, 'text': review_text})

print(f"Total parsed reviews: {len(reviews)}")

# Get unique products
products = sorted(set(r['product'] for r in reviews))
print(f"Unique products: {len(products)}")
print(f"Products: {products}")

# Count reviews per product
product_counts = defaultdict(int)
for r in reviews:
    product_counts[r['product']] += 1
print(f"\nReviews per product:")
for p in sorted(product_counts.keys()):
    print(f"  {p}: {product_counts[p]} reviews")

# --- next REPL turn ---
import asyncio
from collections import defaultdict

# Split reviews into batches for parallel classification
batch_size = 20
batches = [reviews[i:i+batch_size] for i in range(0, len(reviews), batch_size)]

async def classify_batch(batch):
    texts = [r['text'] for r in batch]
    result = await predict(
        "texts: list[str] -> classifications: list[str]",
        instructions="Classify each review as 'positive' or 'negative' based on sentiment. Return a list wit
```

### Baseline on `imdb` (size 1500) failed:

`BadRequestError: litellm.BadRequestError: OpenAIException - request (186981 tokens) exceeds the available context size (65536 tokens), try increasing it`

