#!/usr/bin/env python3
"""Concurrency bake-off for the Paivot multi-agent serving scenario.

Fires C concurrent OpenAI chat-completions at one server. Each request shares a
long common prefix (mimics the RLM system/REPL framing that every ephemeral agent
sends) plus a short unique suffix, then decodes a fixed number of tokens. We sweep
C = 1,2,4,8 and report, per concurrency level:

  - aggregate completion throughput (tokens/s across all in-flight requests)
  - per-request decode tok/s and end-to-end latency (p50/p95)
  - speedup of aggregate throughput vs C=1 (does the engine batch, or serialize?)

The shared prefix is what lets DFlash's prefix cache shine for multi-agent loads;
a serialized engine (no continuous batching) shows flat aggregate throughput as C
grows, while a batching engine (llama.cpp --parallel) scales up.

Stdlib only. Usage:
  python concurrency_bench.py --url http://127.0.0.1:8772/v1 \
      --model mlx-community/Qwen3.6-27B-8bit --label mlx_lm --out results/conc_mlxlm.json
"""
import argparse, json, time, urllib.request, statistics
from concurrent.futures import ThreadPoolExecutor

WORD = "ledger account balance debit credit transaction amount currency posting "

def make_text(approx_tokens: int) -> str:
    # ~1.3 tokens/word for English; generate enough words to hit approx_tokens.
    nwords = int(approx_tokens / 1.3)
    return (WORD * (nwords // 9 + 1))[:nwords * 9]

def one_request(url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "ignore_eos": True, "stream": False,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.monotonic() - t0
    usage = d.get("usage", {}) or {}
    timings = d.get("timings", {}) or {}
    ctoks = usage.get("completion_tokens") or timings.get("predicted_n") or 0
    return {"latency_s": dt, "completion_tokens": ctoks,
            "decode_tps": (ctoks / dt) if dt > 0 else 0}

def run_level(url, model, c, prefix, max_tokens, timeout):
    # C concurrent requests: shared prefix + unique suffix each.
    prompts = [f"{prefix}\n[agent {i}] Continue the running log in detail:" for i in range(c)]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=c) as ex:
        results = list(ex.map(lambda p: one_request(url, model, p, max_tokens, timeout), prompts))
    wall = time.monotonic() - t0
    total_ctoks = sum(r["completion_tokens"] for r in results)
    lats = sorted(r["latency_s"] for r in results)
    def pct(p):
        if not lats: return 0
        k = min(len(lats) - 1, int(round((p / 100) * (len(lats) - 1))))
        return lats[k]
    return {
        "concurrency": c, "wall_s": round(wall, 2),
        "agg_completion_tps": round(total_ctoks / wall, 2) if wall > 0 else 0,
        "total_completion_tokens": total_ctoks,
        "per_req_decode_tps_median": round(statistics.median(r["decode_tps"] for r in results), 2),
        "latency_p50_s": round(pct(50), 2), "latency_p95_s": round(pct(95), 2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--prefix-tokens", type=int, default=5500)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--cooldown", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prefix = make_text(a.prefix_tokens)
    levels = [int(x) for x in a.levels.split(",")]
    # warmup (caches/kernels/prefix)
    print(f"[{a.label}] warmup ...", flush=True)
    one_request(a.url, a.model, prefix + "\nwarmup:", a.max_tokens, a.timeout)
    rows = []
    base = None
    for c in levels:
        time.sleep(a.cooldown)
        r = run_level(a.url, a.model, c, prefix, a.max_tokens, a.timeout)
        if base is None: base = r["agg_completion_tps"] or 1
        r["scale_vs_c1"] = round(r["agg_completion_tps"] / base, 2) if base else 0
        rows.append(r)
        print(f"[{a.label}] C={c:<2} agg_tps={r['agg_completion_tps']:<7} "
              f"scale={r['scale_vs_c1']:<5} per_req_tps={r['per_req_decode_tps_median']:<6} "
              f"p50={r['latency_p50_s']:<6} p95={r['latency_p95_s']}", flush=True)
    out = {"label": a.label, "url": a.url, "model": a.model,
           "prefix_tokens": a.prefix_tokens, "max_tokens": a.max_tokens, "rows": rows}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
