#!/usr/bin/env python3
"""Smoke-test the MultiPL-E execution service: run one known-good program per language.

Proves every core-6 compiler/runtime works inside the (aarch64) image. Uses the vendored
sample_payloads/ where available and an inline trivial program for bash.

  python smoke_test.py --url http://127.0.0.1:8123
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "sample_payloads"

# Vendored sample payloads that are known-good completions (label, payload-file).
CASES = [
    ("java", "payload_java.json"),
    ("javascript", "payload_js.json"),
    ("php", "payload_php.json"),
    ("rust", "payload_rust.json"),
]

# Inline trivial known-good programs for cpp and bash. (The vendored payload_cpp.json is an
# imperfect completion that intentionally fails to compile, so we use our own here.)
INLINE = [
    {
        "res_id": "cpp_smoke",
        "language": "cpp",
        "completion": "#include <cassert>\nint add(int a, int b) { return a + b; }\n",
        "test": "int main() { assert(add(2, 3) == 5); return 0; }\n",
    },
    {
        "res_id": "sh_smoke",
        "language": "sh",
        "completion": "add() {\n  echo $(( $1 + $2 ))\n}\n",
        "test": 'result=$(add 2 3)\nif [ "$result" != "5" ]; then exit 1; fi\n',
    },
    {
        "res_id": "rb_smoke",
        "language": "rb",
        "completion": "def add(a, b)\n  a + b\nend\n",
        "test": "raise 'fail' unless add(2, 3) == 5\n",
    },
    {
        "res_id": "ts_smoke",
        "language": "ts",
        "completion": "function add(a: number, b: number): number {\n  return a + b;\n}\n",
        "test": "if (add(2, 3) !== 5) {\n  throw new Error('fail');\n}\n",
    },
    {
        # go's dataset language is "go_test.go"; the program must be a valid *_test.go file.
        "res_id": "go_smoke",
        "language": "go_test.go",
        "completion": 'package main\n\nimport "testing"\n\nfunc add(a int, b int) int {\n  return a + b\n}\n',
        "test": 'func TestAdd(t *testing.T) {\n  if add(2, 3) != 5 {\n    t.Fatalf("want 5")\n  }\n}\n',
    },
]


def post(url: str, payload: dict) -> dict:
    # The execution service requires res_id to be a string (pydantic v2 won't coerce
    # int->str); some vendored sample payloads use ints, so normalize here.
    payload = {**payload, "res_id": str(payload["res_id"])}
    r = httpx.post(url.rstrip("/") + "/execute", json=payload, timeout=60.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()

    payloads = []
    for label, fname in CASES:
        payloads.append((label, json.loads((SAMPLES / fname).read_text())))
    for payload in INLINE:
        payloads.append((payload["language"], payload))

    all_ok = True
    print(f"{'language':<12} {'passed':<7} exec_result")
    print("-" * 40)
    for label, payload in payloads:
        try:
            res = post(args.url, payload)
            passed = int(res.get("passed", 0))
            note = res.get("exec_result", "")
        except Exception as e:  # noqa: BLE001
            passed, note = 0, f"RequestError: {e}"
        ok = passed == 1
        all_ok = all_ok and ok
        flag = "OK " if ok else "FAIL"
        print(f"{label:<12} {flag:<7} {note}")

    print("-" * 40)
    print("ALL PASSED" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
