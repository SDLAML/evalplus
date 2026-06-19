#!/usr/bin/env python3
"""Benchmark the MultiPL-E executor against real generated completions.

Reports, per concurrency level: wall time, throughput, per-request latency percentiles, and
the status mix (OK / SyntaxError / Exception / Timeout / RunnerError). Use it to see whether
"verify" time is dominated by the per-test compile/link floor, by hanging tests (Timeout), by
dropped requests (RunnerError = server overload), or simply by core starvation.

  python bench_exec.py --url http://127.0.0.1:8123 \
      --file .../multipl_e/mbpp-go/gen/mbpp-go.jsonl --limit 1000 --concurrency 16,48,96,192
"""
import argparse, json, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx


def load(path, limit):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return recs[:limit] if limit else recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", default="16,48,96,192")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    recs = load(args.file, args.limit)
    exurl = args.url.rstrip("/") + "/execute"
    print(f"file={args.file}\nrecords={len(recs)} timeout={args.timeout}s")

    def one(rec):
        payload = {"res_id": rec["res_id"], "language": rec["language"],
                   "completion": rec["completion"], "test": rec["test"], "timeout": args.timeout}
        t = time.time()
        try:
            r = httpx.post(exurl, json=payload, timeout=args.timeout * 3)
            r.raise_for_status()
            st = r.json().get("exec_result", "?")
        except Exception as e:  # noqa: BLE001
            st = f"RunnerError:{type(e).__name__}"
        return st, time.time() - t

    # warm the go build cache so we measure steady state
    for rec in recs[:8]:
        one(rec)

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    for C in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=C) as ex:
            out = list(ex.map(one, recs))
        dt = time.time() - t0
        stats = Counter(s for s, _ in out)
        lat = [d for _, d in out]
        print(f"\n--- concurrency={C} ---")
        print(f"  wall={dt:7.1f}s  throughput={len(recs)/dt:6.1f}/s  "
              f"latency s: median={pct(lat,0.5):.2f} p90={pct(lat,0.9):.2f} max={max(lat):.2f}")
        print(f"  status: " + "  ".join(f"{k}={v}" for k, v in stats.most_common()))


if __name__ == "__main__":
    main()
