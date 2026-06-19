import json

import anyio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from eval import check_correctness


app = FastAPI(title="MultiPL Execution Service", version="1.0.0")


@app.on_event("startup")
async def _configure_threadpool() -> None:
    # /execute is a *sync* endpoint, so Starlette runs each call in a worker thread; raise
    # the per-process thread limit so one uvicorn worker can run many compiles concurrently
    # (real parallelism is bounded by CPUs and by the client's --n-workers).
    anyio.to_thread.current_default_thread_limiter().total_tokens = 128


class MultiPLExecutionRequest(BaseModel):
    res_id: str
    language: str
    completion: str
    test: str
    # timeout: float = 10.0  # TODO: pipe this through to safe_subprocess.run


class MultiPLExecutionResponse(BaseModel):
    res_id: str
    tested_completion: str
    passed: int
    exec_result: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/execute", response_model=MultiPLExecutionResponse)
def execute_code(request: MultiPLExecutionRequest):
    # NOTE: sync def on purpose. check_correctness() blocks (compiles/runs a subprocess);
    # a sync endpoint runs in Starlette's threadpool so concurrent requests run in parallel
    # instead of serializing on the event loop (the old `async def` capped each worker to 1).
    try:
        result = check_correctness(request.language, request.completion, request.test)
        passed = 1 if result["status"] == "OK" and result["exit_code"] == 0 else 0

        return MultiPLExecutionResponse(
            res_id=request.res_id,
            tested_completion=result["tested_completion"],
            passed=passed,
            exec_result=result["status"],
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Handler for AWS Lambda
def handler(event, context):
    request = MultiPLExecutionRequest(**event)
    response = execute_code(request)
    return json.dumps(response.dict())
