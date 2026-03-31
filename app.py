from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from opsflow_env.env import OpsFlowEnv
from opsflow_env.models import OpsFlowAction


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: OpsFlowAction


env = OpsFlowEnv()
app = FastAPI(title="OpsFlowEnv", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset_endpoint(request: Optional[ResetRequest] = None) -> dict:
    task_id = request.task_id if request is not None else None
    result = env.reset(task_id=task_id)
    return result.model_dump(mode="json")


@app.post("/step")
def step_endpoint(request: StepRequest) -> dict:
    result = env.step(request.action)
    return result.model_dump(mode="json")


@app.get("/state")
def state_endpoint() -> dict:
    return env.state().model_dump(mode="json")


def main() -> None:
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
