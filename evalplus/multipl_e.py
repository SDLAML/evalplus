#!/usr/bin/env python3
"""MultiPL-E runner (HumanEval / MBPP for cpp/java/js/php/rs/sh/go/rb/ts).

Lives in the evalplus package so it can be driven uniformly alongside native evalplus:
`humaneval`/`mbpp` use evalplus's Python sandbox, while `humaneval-<lang>`/`mbpp-<lang>`
route here (see runner_lighteval/lib_eval_jobs.sh). Invoke as `python -m evalplus.multipl_e`.

Subcommands:
  run       generate + execute for a single `<bench>-<lang>` dataset (one-shot; mirrors
            evalplus's one-command flow). This is what lib_eval_jobs.sh calls.
  generate  only generate completions to JSONL (decoupled, resume-aware).
  execute   only execute existing completions against the running executor service.
  prefetch  download nuprl/MultiPL-E configs into the HF cache (run where there's internet).

Generation is base/completion style: feed the dataset `prompt` to a vLLM OpenAI-compatible
*completions* endpoint with per-row stop tokens, then `completion = prompt + continuation`.
(evalplus's OpenAIChatDecoder is unusable here: it wraps prompts in ``` fences, hardcodes a
```python``` response prefix, and passes no stop tokens.) Default sampling is greedy pass@1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# Config suffixes in nuprl/MultiPL-E (humaneval-<lang> / mbpp-<lang>). The per-row
# `language` field may differ (e.g. go's is "go_test.go") and is read from the dataset.
SUPPORTED_LANGS = ["cpp", "java", "js", "php", "rs", "sh", "go", "rb", "ts"]
BENCHMARKS = ["humaneval", "mbpp"]
DEFAULT_PASS_AT_K = (1, 10, 100)


# --------------------------------------------------------------------------------------
# pass@k (standard unbiased estimator from the HumanEval paper)
# --------------------------------------------------------------------------------------
def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Probability that at least one of k draws (without replacement) is correct."""
    import numpy as np

    if num_samples - num_correct < k:
        return 1.0
    return float(
        1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))
    )


# --------------------------------------------------------------------------------------
# selection helpers
# --------------------------------------------------------------------------------------
def resolve_langs(langs: str) -> List[str]:
    if langs.strip().lower() == "all":
        return list(SUPPORTED_LANGS)
    out = [x.strip() for x in langs.split(",") if x.strip()]
    bad = [x for x in out if x not in SUPPORTED_LANGS]
    if bad:
        sys.exit(f"Unsupported language(s) {bad}; choose from {SUPPORTED_LANGS}")
    return out


def resolve_benches(benchmark: str) -> List[str]:
    if benchmark == "both":
        return list(BENCHMARKS)
    if benchmark not in BENCHMARKS:
        sys.exit(f"Unknown benchmark {benchmark!r}; choose from {BENCHMARKS + ['both']}")
    return [benchmark]


def parse_dataset(name: str) -> Tuple[str, str]:
    """Split a `<bench>-<lang>` dataset name, e.g. 'humaneval-js' -> ('humaneval', 'js')."""
    for bench in BENCHMARKS:
        prefix = f"{bench}-"
        if name.startswith(prefix):
            lang = name[len(prefix):]
            if lang not in SUPPORTED_LANGS:
                sys.exit(f"Unsupported language {lang!r} in dataset {name!r}; choose {SUPPORTED_LANGS}")
            return bench, lang
    sys.exit(f"Dataset {name!r} is not a MultiPL-E task; expected '<humaneval|mbpp>-<lang>'")


def gen_file(output_dir: Path, bench: str, lang: str) -> Path:
    return output_dir / f"{bench}-{lang}.jsonl"


def load_dataset_rows(bench: str, lang: str):
    """Load one nuprl/MultiPL-E config; respects HF_HUB_OFFLINE on compute nodes."""
    from datasets import load_dataset

    return load_dataset("nuprl/MultiPL-E", f"{bench}-{lang}", split="test")


def resolve_sampling(args: argparse.Namespace) -> Tuple[int, float]:
    """Return (n_samples, temperature). Default is greedy pass@1 (matches evalplus --greedy)."""
    if getattr(args, "greedy", False):
        return 1, 0.0
    if getattr(args, "temperature", None) is not None:
        if args.temperature <= 0:
            sys.exit("--temperature must be > 0 (use --greedy for deterministic decoding)")
        return args.n_samples, args.temperature
    # Neither --greedy nor --temperature given -> default to greedy pass@1.
    return 1, 0.0


def resolve_pass_at_k(args: argparse.Namespace, n_samples: Optional[int] = None) -> List[int]:
    """Match native EvalPlus pass@k defaults unless the user explicitly overrides them."""
    if getattr(args, "pass_at_k", None):
        return [int(x) for x in str(args.pass_at_k).split(",") if str(x).strip()]
    if n_samples is None:
        return list(DEFAULT_PASS_AT_K)
    return [k for k in DEFAULT_PASS_AT_K if k <= n_samples] or [1]


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------
def _count_existing(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = json.loads(line)["name"]
            counts[name] = counts.get(name, 0) + 1
    return counts


def _complete(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    stop: Optional[List[str]],
    retries: int = 3,
) -> List[str]:
    """Call the OpenAI-compatible /completions endpoint; return n continuations."""
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if temperature > 0:
        payload["top_p"] = 0.95
    if stop:
        payload["stop"] = stop  # continuations exclude the stop sequence
    url = base_url.rstrip("/") + "/completions"
    last_err = None
    for attempt in range(retries):
        try:
            r = client.post(url, json=payload)
            r.raise_for_status()
            choices = r.json()["choices"]
            return [c.get("text", "") for c in choices]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"completion request failed after {retries} tries: {last_err}")


def generate_one(
    client: httpx.Client,
    base_url: str,
    model: str,
    bench: str,
    lang: str,
    out: Path,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    gen_workers: int,
    resume: bool,
    limit: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = _count_existing(out) if resume else {}
    rows = load_dataset_rows(bench, lang)
    if limit:
        rows = rows.select(range(min(limit, len(rows))))

    jobs = []
    for row in rows:
        need = n_samples - existing.get(row["name"], 0)
        if need > 0:
            jobs.append((row, existing.get(row["name"], 0), need))

    done = sum(existing.get(r["name"], 0) for r in rows)
    total = len(rows) * n_samples
    print(f"[generate] {bench}-{lang}: {len(rows)} problems, {n_samples} samples each, "
          f"{done}/{total} already present")
    if not jobs:
        return

    def _run(job):
        row, have, need = job
        texts = _complete(
            client, base_url, model,
            prompt=row["prompt"], n=need, temperature=temperature,
            max_tokens=max_tokens, stop=list(row["stop_tokens"]),
        )
        out_lines = []
        for j, text in enumerate(texts):
            out_lines.append({
                "res_id": f"{row['name']}::{have + j}",
                "name": row["name"],
                "language": row["language"],
                "prompt": row["prompt"],
                # eval.py runs `completion + "\n" + test`, so completion is prompt + continuation.
                "completion": row["prompt"] + text,
                "test": row["tests"],
            })
        return out_lines

    written = 0
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=gen_workers) as ex:
        for fut in as_completed([ex.submit(_run, job) for job in jobs]):
            for rec in fut.result():
                fh.write(json.dumps(rec) + "\n")
                written += 1
            fh.flush()
    print(f"[generate] {bench}-{lang}: wrote {written} new samples -> {out}")


def cmd_generate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    n_samples, temperature = resolve_sampling(args)
    benches = resolve_benches(args.benchmark)
    langs = resolve_langs(args.langs)
    with httpx.Client(timeout=httpx.Timeout(args.http_timeout, connect=30.0)) as client:
        for bench in benches:
            for lang in langs:
                generate_one(
                    client, args.base_url, args.model, bench, lang,
                    gen_file(output_dir, bench, lang),
                    n_samples, temperature, args.max_tokens, args.gen_workers,
                    args.resume, args.limit,
                )


# --------------------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------------------
def _wait_healthy(exec_url: str, timeout_s: float = 120.0) -> None:
    url = exec_url.rstrip("/") + "/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    sys.exit(f"executor at {exec_url} not healthy after {timeout_s:.0f}s")


def _execute_one(client: httpx.Client, exec_url: str, rec: dict, timeout: int,
                 retries: int = 3) -> dict:
    url = exec_url.rstrip("/") + "/execute"
    payload = {
        "res_id": rec["res_id"],
        "language": rec["language"],
        "completion": rec["completion"],
        "test": rec["test"],
        "timeout": timeout,
    }
    # Retry transient transport/HTTP failures (e.g. server briefly overloaded under high
    # concurrency). Without this, a dropped response is silently scored as passed=0, which
    # would corrupt pass@k. A real execution result (incl. Timeout) returns 200 and is kept.
    last_err = None
    for attempt in range(retries):
        try:
            r = client.post(url, json=payload)
            r.raise_for_status()
            body = r.json()
            return {"res_id": rec["res_id"], "name": rec["name"],
                    "passed": int(body.get("passed", 0)), "exec_result": body.get("exec_result", "")}
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return {"res_id": rec["res_id"], "name": rec["name"],
            "passed": 0, "exec_result": f"RunnerError: {last_err}"}


def execute_file(
    client: httpx.Client,
    exec_url: str,
    bench: str,
    lang: str,
    recs: List[dict],
    n_workers: int,
    timeout: int,
    ks: List[int],
) -> dict:
    outcomes: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_execute_one, client, exec_url, rec, timeout) for rec in recs]
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            outcomes[res["res_id"]] = res
            done += 1
            if done % 200 == 0 or done == len(futures):
                print(f"[execute] {bench}-{lang}: {done}/{len(futures)}")

    per_problem: Dict[str, Dict[str, int]] = {}
    for res in outcomes.values():
        p = per_problem.setdefault(res["name"], {"n": 0, "c": 0})
        p["n"] += 1
        p["c"] += int(res["passed"])

    pass_at_k = {}
    for k in ks:
        vals = [estimate_pass_at_k(p["n"], p["c"], k) for p in per_problem.values() if p["n"] >= k]
        if vals:
            pass_at_k[f"pass@{k}"] = sum(vals) / len(vals)

    return {
        "benchmark": bench,
        "language": lang,
        "task": f"multipl_e_{bench}:{lang}",
        "n_problems": len(per_problem),
        "n_samples_total": len(outcomes),
        "pass_at_k": pass_at_k,
        "per_problem": per_problem,
    }


def cmd_execute(args: argparse.Namespace) -> None:
    gen_dir = Path(args.generations_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ks = resolve_pass_at_k(args)
    benches = resolve_benches(args.benchmark)
    langs = resolve_langs(args.langs)

    _wait_healthy(args.exec_url)
    http_timeout = httpx.Timeout(args.timeout * 3 + 30.0, connect=30.0)
    summary = {}
    with httpx.Client(timeout=http_timeout) as client:
        for bench in benches:
            for lang in langs:
                path = gen_file(gen_dir, bench, lang)
                if not path.exists():
                    print(f"[execute] {bench}-{lang}: no generations at {path}, skipping")
                    continue
                recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                if not recs:
                    continue
                result = execute_file(client, args.exec_url, bench, lang, recs,
                                      args.n_workers, args.timeout, ks)
                out_path = results_dir / f"{bench}-{lang}.json"
                out_path.write_text(json.dumps(result, indent=2))
                summary[result["task"]] = result["pass_at_k"]
                pk = "  ".join(f"{k}={v:.4f}" for k, v in result["pass_at_k"].items())
                print(f"[execute] {bench}-{lang}: {result['n_problems']} problems  {pk}  -> {out_path}")

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(summary, results_dir / "summary.json")


# --------------------------------------------------------------------------------------
# run (generate + execute for one <bench>-<lang>, the lib_eval_jobs.sh entrypoint)
# --------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> None:
    bench, lang = parse_dataset(args.dataset)
    n_samples, temperature = resolve_sampling(args)
    ks = resolve_pass_at_k(args, n_samples)

    base = Path(args.root) / "multipl_e" / args.dataset
    gen_dir = base / "gen"
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = gen_file(gen_dir, bench, lang)

    print(f"[multipl_e] dataset={args.dataset} samples={n_samples} temp={temperature} root={base}")
    with httpx.Client(timeout=httpx.Timeout(args.http_timeout, connect=30.0)) as client:
        generate_one(client, args.base_url, args.model, bench, lang, out,
                     n_samples, temperature, args.max_tokens, args.gen_workers,
                     args.resume, args.limit)

    _wait_healthy(args.exec_url)
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    http_timeout = httpx.Timeout(args.timeout * 3 + 30.0, connect=30.0)
    with httpx.Client(timeout=http_timeout) as client:
        result = execute_file(client, args.exec_url, bench, lang, recs,
                              args.n_workers, args.timeout, ks)

    (results_dir / f"{bench}-{lang}.json").write_text(json.dumps(result, indent=2))
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(json.dumps(result, indent=2))
    summary = {result["task"]: result["pass_at_k"]}
    _print_summary(summary, results_dir / f"{bench}-{lang}.json")


# --------------------------------------------------------------------------------------
# prefetch
# --------------------------------------------------------------------------------------
def cmd_prefetch(args: argparse.Namespace) -> None:
    for bench in resolve_benches(args.benchmark):
        for lang in resolve_langs(args.langs):
            ds = load_dataset_rows(bench, lang)
            print(f"[prefetch] cached {bench}-{lang}: {len(ds)} rows")
    print("Done. Compute nodes can now run with HF_HUB_OFFLINE=1.")


def _print_summary(summary: dict, path: Path) -> None:
    print("\n=== MultiPL-E summary ===")
    for task, pk in summary.items():
        print(f"  {task}: " + "  ".join(f"{k}={v:.4f}" for k, v in pk.items()))
    print(f"\nWrote {path}")


# --------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------
def _add_sampling_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--greedy", action="store_true", help="temperature 0, 1 sample/problem (default)")
    p.add_argument("--temperature", type=float, default=None, help="enable sampling at this temp")
    p.add_argument("--n-samples", "--n_samples", dest="n_samples", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="cap problems per task (0 = all)")
    p.add_argument("--gen-workers", type=int, default=16)
    p.add_argument("--http-timeout", type=float, default=600.0)
    p.add_argument("--trust-remote-code", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--thinking-mode", choices=["on", "off"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--override-chat-template", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-override-chat-template", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)


def _add_exec_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-workers", type=int,
                   default=int(os.environ.get("MULTIPLE_EXEC_CONCURRENCY", "0")) or (os.cpu_count() or 48),
                   help="concurrent /execute requests (CPU-bound; size ~= allocated cores)")
    p.add_argument("--timeout", type=int, default=30, help="per-execution timeout (s)")
    p.add_argument("--pass-at-k", default=None, help="comma list, e.g. 1,10")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evalplus.multipl_e", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate + execute one <bench>-<lang> dataset")
    r.add_argument("--dataset", required=True, help="e.g. humaneval-js, mbpp-cpp")
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", required=True, help="vLLM OpenAI base, e.g. http://127.0.0.1:8000/v1")
    r.add_argument("--exec-url", required=True, help="executor base, e.g. http://127.0.0.1:8123")
    r.add_argument("--root", required=True, help="output root (writes <root>/multipl_e/<dataset>/)")
    r.add_argument("--output-file", default="", help="also write the results json here")
    _add_sampling_flags(r)
    _add_exec_flags(r)
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("generate", help="generate completions via a vLLM /v1 endpoint")
    g.add_argument("--benchmark", default="both", choices=["humaneval", "mbpp", "both"])
    g.add_argument("--langs", default="all", help="comma list of cpp,java,js,php,rs,sh or 'all'")
    g.add_argument("--model", required=True)
    g.add_argument("--base-url", required=True)
    g.add_argument("--output-dir", required=True)
    _add_sampling_flags(g)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("execute", help="execute completions against the MultiPL-E service")
    e.add_argument("--benchmark", default="both", choices=["humaneval", "mbpp", "both"])
    e.add_argument("--langs", default="all", help="comma list or 'all'")
    e.add_argument("--exec-url", required=True)
    e.add_argument("--generations-dir", required=True)
    e.add_argument("--results-dir", required=True)
    _add_exec_flags(e)
    e.set_defaults(func=cmd_execute)

    pf = sub.add_parser("prefetch", help="seed the HF cache (run where there's internet)")
    pf.add_argument("--benchmark", default="both", choices=["humaneval", "mbpp", "both"])
    pf.add_argument("--langs", default="all")
    pf.set_defaults(func=cmd_prefetch)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
