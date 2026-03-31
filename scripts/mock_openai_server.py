from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="OpsFlow Mock OpenAI")


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None


def scripted_action(observation: dict[str, Any]) -> dict[str, Any]:
    task_id = observation["task_id"]
    step = observation["step_count"]

    plans: dict[str, dict[int, dict[str, Any]]] = {
        "easy_single_request_clear_slot": {
            0: {"action_type": "open_thread", "thread_id": "t_easy_1"},
            1: {"action_type": "classify_thread", "category": "meeting_request"},
            2: {"action_type": "set_priority", "priority": "high"},
            3: {
                "action_type": "extract_constraints",
                "extracted_fields": {
                    "request_type": "schedule_meeting",
                    "attendees": ["riya", "aman"],
                    "duration_minutes": 30,
                    "preferred_dates": ["2026-04-02"],
                    "preferred_time_ranges": ["afternoon"],
                    "timezone": "Asia/Calcutta",
                    "missing_fields": [],
                },
            },
            4: {"action_type": "view_calendar", "participant_ids": ["riya", "aman"]},
            5: {
                "action_type": "book_meeting",
                "proposed_start": "2026-04-02T15:00:00+05:30",
                "participant_ids": ["riya", "aman"],
                "duration_minutes": 30,
            },
        },
        "medium_missing_info_with_conflict": {
            0: {"action_type": "open_thread", "thread_id": "t_medium_1"},
            1: {"action_type": "classify_thread", "category": "meeting_request"},
            2: {"action_type": "set_priority", "priority": "medium"},
            3: {
                "action_type": "extract_constraints",
                "extracted_fields": {
                    "request_type": "schedule_meeting",
                    "attendees": ["priya", "karan"],
                    "preferred_dates": ["2026-04-04"],
                    "preferred_time_ranges": ["morning"],
                    "timezone": "Asia/Calcutta",
                    "missing_fields": ["duration_minutes"],
                },
            },
            4: {
                "action_type": "propose_slot",
                "proposed_start": "2026-04-04T10:00:00+05:30",
                "participant_ids": ["priya", "karan"],
                "duration_minutes": 30,
            },
            5: {"action_type": "ask_for_missing_info", "reason": "duration_minutes missing"},
        },
        "hard_priority_conflict_timezone": {
            0: {"action_type": "open_thread", "thread_id": "t_hard_1"},
            1: {"action_type": "classify_thread", "category": "meeting_request"},
            2: {"action_type": "set_priority", "priority": "high"},
            3: {
                "action_type": "extract_constraints",
                "extracted_fields": {
                    "request_type": "schedule_meeting",
                    "attendees": ["vp_lex", "maya_ops", "jun_pm"],
                    "duration_minutes": 30,
                    "preferred_dates": ["2026-04-07"],
                    "preferred_time_ranges": ["before_noon_singapore"],
                    "timezone": "Asia/Singapore",
                    "deadline": "2026-04-07T12:00:00+08:00",
                    "missing_fields": [],
                },
            },
            4: {"action_type": "view_calendar", "participant_ids": ["vp_lex", "maya_ops", "jun_pm"]},
            5: {
                "action_type": "book_meeting",
                "proposed_start": "2026-04-07T11:00:00+08:00",
                "participant_ids": ["vp_lex", "maya_ops", "jun_pm"],
                "duration_minutes": 30,
            },
        },
    }

    return plans.get(task_id, {}).get(step, {"action_type": "finish", "reason": "no scripted action"})


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionsRequest) -> dict[str, Any]:
    user_message = request.messages[-1]["content"]
    observation = json.loads(user_message)
    action = scripted_action(observation)
    return {
        "id": "chatcmpl-opsflow-mock",
        "object": "chat.completion",
        "created": 0,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(action)},
                "finish_reason": "stop",
            }
        ],
    }


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8010)


if __name__ == "__main__":
    main()
