# MultiPL-E code-execution service (Apptainer / aarch64)

Vendored from [olmes-docker `src/multiple_execution`](https://github.com/allenai/olmes-docker)
(itself adapted from [MultiPL-E](https://github.com/nuprl/MultiPL-E)). This is the sandbox
that compiles and runs model-generated code for the MultiPL-E HumanEval/MBPP tasks, served
as a small FastAPI app. We package it as an **Apptainer** image for our HPC GH200
(linux-aarch64) nodes — the upstream Dockerfile targeted an x86 AWS-Lambda base.

Driven by [`runner_lighteval/run_multipl_e.py`](../../runner_lighteval/run_multipl_e.py).

## Languages

Executed inside the image (9): **cpp, java, js, php, rs (rust), sh (bash), go, rb (ruby),
ts (typescript)** (plus python). Note Go's dataset `language` field is `go_test.go` and uses
`go test` (the temp file must end in `_test.go`). The remaining `eval_*.py` files (cs/swift/…)
are carried along but their toolchains are not installed.

## HTTP contract

- `GET /health` → `{"status": "healthy"}`
- `POST /execute` with `{"res_id": str, "language": str, "completion": str, "test": str}`
  → `{"res_id", "tested_completion", "passed": 0|1, "exec_result": str}`

The service runs `program = completion + "\n" + test`, so `completion` must be the full
program *minus* the tests (i.e. the dataset `prompt` + the model continuation). `res_id`
**must be a string** (pydantic v2 won't coerce ints).

## Build (on a node WITH internet, e.g. login node)

```sh
bash build_executor.sh                 # -> ./multipl-exec.sif (aarch64)
# build scratch must be a LOCAL fs (squashfs packing fails on Lustre/scratch):
#   APPTAINER_TMPDIR=/tmp/apptainer-$USER bash build_executor.sh
```

The image bakes in all toolchains, so it needs **no network at run time**.

## Run the service (compute node, offline)

```sh
# Always use --cleanenv so an activated host venv can't leak into the container python.
APPTAINERENV_PORT=8123 APPTAINERENV_WORKERS=8 \
  apptainer run --cleanenv multipl-exec.sif &
curl -s http://127.0.0.1:8123/health        # {"status":"healthy"}
```

`WORKERS` sets the number of uvicorn workers (parallel compiles); match it to the client
concurrency (`run_multipl_e.py execute --n-workers`). Code is compiled/run under `/tmp`
(writable in Apptainer); no GPU and no network are required.

## Smoke test (one known-good program per language)

```sh
python smoke_test.py --url http://127.0.0.1:8123   # expects ALL PASSED
```

## Driving it

The logic lives in the **`evalplus.multipl_e`** module (`python -m evalplus.multipl_e`,
with the thin `runner_lighteval/run_multipl_e.py` wrapper). Subcommands: `run` (generate +
execute one `<bench>-<lang>`), `generate`, `execute`, `prefetch`. Default sampling = greedy
pass@1.

One-shot for a single dataset (what the SLURM dispatch calls):
```sh
python -m evalplus.multipl_e run \
    --dataset humaneval-js --model <served-name> \
    --base-url http://127.0.0.1:8000/v1 --exec-url http://127.0.0.1:8123 \
    --root runs/<model>            # writes runs/<model>/multipl_e/humaneval-js/{gen,results}/
```

Decoupled (generate all, then execute all):
```sh
python -m evalplus.multipl_e generate --benchmark both --langs all --greedy \
    --model <served-name> --base-url http://127.0.0.1:8000/v1 --output-dir runs/<model>/gen
python -m evalplus.multipl_e execute --exec-url http://127.0.0.1:8123 \
    --generations-dir runs/<model>/gen --results-dir runs/<model>/results --pass-at-k 1
```

### Via the eval pipeline (`lib_eval_jobs.sh`)

List MultiPL-E tasks right alongside the Python ones in `EVALPLUS_DATASETS`; the dispatch
routes `humaneval`/`mbpp` to native evalplus and `humaneval-<lang>`/`mbpp-<lang>` here, and
starts/stops this executor service automatically (requires `BACKEND_MODE=container_vllm`):
```bash
EVALPLUS_DATASETS=( "humaneval" "mbpp" "humaneval-js" "mbpp-cpp" "humaneval-rs" )
EXTRA_MULTIPLE_ARGS=(--pass-at-k 1)        # optional MultiPL-E knobs; greedy pass@1 default
# Optional overrides: MULTIPLE_EXEC_SIF, MULTIPLE_EXEC_PORT (8123), MULTIPLE_EXEC_WORKERS (8)
```

## Notes / gotchas

- `multipl-exec.def` reproduces two paths hardcoded by the evaluators: the JavaTuples jar
  at `/opt/lib/javatuples-1.2.jar` (`eval_java.py`) and `rustc` at `/opt/bin/rustc`
  (`eval_rust.py`, invoked without env → `%environment` exports `CARGO_HOME`/`RUSTUP_HOME`).
- Use the **headless** JDK in the def; the full `default-jdk` pulls `systemd`, whose
  post-install script breaks fakeroot/proot builds.

---

## Appendix: upstream AWS-Lambda deployment (not used here)

The original olmes-docker deployment (kept for reference):

```sh
docker buildx build --platform linux/amd64 -t codex/multiple_ex:latest .
aws lambda update-function-code --function-name multiple-ex \
  --image-uri <acct>.dkr.ecr.us-east-1.amazonaws.com/codex/multiple_ex:latest --region us-east-1
```
