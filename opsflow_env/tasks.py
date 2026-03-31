from __future__ import annotations

import json
from pathlib import Path

from opsflow_env.models import TaskDefinition


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_task(task_id: str) -> TaskDefinition:
    fixture_path = FIXTURES_DIR / f"{task_id}.json"
    if not fixture_path.exists():
        raise KeyError(f"Unknown task_id: {task_id}")
    return TaskDefinition.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def load_all_tasks() -> list[TaskDefinition]:
    tasks: list[TaskDefinition] = []
    for fixture_path in sorted(FIXTURES_DIR.glob("*.json")):
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        tasks.append(TaskDefinition.model_validate(payload))
    return tasks
