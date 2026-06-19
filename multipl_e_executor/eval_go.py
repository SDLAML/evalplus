import os
import subprocess
from pathlib import Path


def eval_script(path: Path):
    status = None
    stdout = None
    stderr = None
    exit_code = None
    try:
        # `-timeout 10s`: model generations frequently contain infinite loops; this makes
        # `go test` self-terminate a hung test (panic: test timed out -> FAIL) at ~10s
        # instead of riding the full subprocess timeout. The subprocess timeout is the
        # outer backstop in case `go` itself wedges before its own timeout fires.
        # -vet=off skips the static vet pass (pure compile+run speedup, no effect on result).
        # GOMAXPROCS=1 + -p=1: by default each `go test` grabs ALL cores for its build, so
        # running many in parallel (our model) spawns ~ncpu*concurrency workers and thrashes
        # the scheduler. Pinning each invocation to 1 core and letting the SERVER provide
        # the parallelism (many concurrent /execute) ~doubles aggregate Go throughput.
        env = dict(os.environ, GOMAXPROCS="1")
        build = subprocess.run(
            ["go", "test", "-vet=off", "-p=1", "-timeout", "10s", str(path)],
            timeout=30,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        stdout = build.stdout.decode("utf-8", errors="ignore")
        stderr = build.stderr.decode("utf-8", errors="ignore")
        exit_code = build.returncode

        # `go test` prints the "[build failed]" summary line to stdout and the compiler
        # diagnostics to stderr, so scan both. Use the exit code (not a "FAIL" substring)
        # to separate a clean pass from a failed/panicked/timed-out test.
        blob = stdout + stderr
        if "[setup failed]" in blob or "[build failed]" in blob:
            status = "SyntaxError"            # did not compile
        elif "no test files" in blob or "no tests to run" in blob:
            status = "SyntaxError"            # nothing ran -> treat as a failure, not a pass
        elif build.returncode == 0:
            status = "OK"                     # all tests passed
        else:
            status = "Exception"              # test failed / panicked / timed out
    except subprocess.TimeoutExpired:
        status = "Timeout"
    except FileNotFoundError:
        # `go` toolchain missing from the image; surface it rather than crashing the worker.
        status = "SyntaxError"
        stderr = "go toolchain not found on PATH"
        exit_code = -1

    return {
        "status": status,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }
