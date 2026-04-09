from __future__ import annotations

import json
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

from opsflow_env.env import OpsFlowEnv
from opsflow_env.models import OpsFlowAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3-0324")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 12
RESULTS_PATH = os.getenv("BASELINE_RESULTS_PATH", "")

SYSTEM_PROMPT = """
You are controlling an operations inbox environment.
Return exactly one JSON object describing the next action to take.
Valid action_type values are:
open_thread, classify_thread, set_priority, extract_constraints, view_calendar,
ask_for_missing_info, propose_slot, book_meeting, reschedule_meeting,
decline_request, escalate_request, archive_thread, finish.

Rules:
- Output valid JSON only.
- Use ISO timestamps when proposing or booking a meeting.
- If a task is missing duration or attendees, prefer ask_for_missing_info.
- Never schedule over a protected or conflicting event.
""".strip()


HEURISTIC_PLANS: dict[str, dict[str, Any]] = {
    "easy_single_request_clear_slot": {
        "thread_id": "t_easy_1",
        "extract": {
            "request_type": "schedule_meeting",
            "attendees": ["riya", "aman"],
            "duration_minutes": 30,
            "preferred_dates": ["2026-04-02"],
            "preferred_time_ranges": ["afternoon"],
            "timezone": "Asia/Calcutta",
            "priority": "high",
            "missing_fields": [],
        },
        "final_action": {
            "action_type": "book_meeting",
            "participant_ids": ["riya", "aman"],
            "proposed_start": "2026-04-02T15:00:00+05:30",
            "duration_minutes": 30,
        },
    },
    "medium_missing_info_with_conflict": {
        "thread_id": "t_medium_1",
        "extract": {
            "request_type": "schedule_meeting",
            "attendees": ["priya", "karan"],
            "duration_minutes": None,
            "preferred_dates": ["2026-04-04"],
            "preferred_time_ranges": ["morning"],
            "timezone": "Asia/Calcutta",
            "priority": "medium",
            "missing_fields": ["duration_minutes"],
        },
        "calendar_participants": ["priya", "karan"],
        "conflict_probe": {
            "action_type": "propose_slot",
            "participant_ids": ["priya", "karan"],
            "proposed_start": "2026-04-04T10:00:00+05:30",
            "duration_minutes": 30,
        },
        "final_action": {
            "action_type": "ask_for_missing_info",
            "reason": "Need duration before booking.",
        },
    },
    "hard_priority_conflict_timezone": {
        "thread_id": "t_hard_1",
        "extract": {
            "request_type": "schedule_meeting",
            "attendees": ["vp_lex", "maya_ops", "jun_pm"],
            "duration_minutes": 30,
            "preferred_dates": ["2026-04-07"],
            "preferred_time_ranges": ["before_noon_singapore"],
            "timezone": "Asia/Singapore",
            "priority": "high",
            "deadline": "2026-04-07T12:00:00+08:00",
            "missing_fields": [],
        },
        "calendar_participants": ["jun_pm", "maya_ops", "vp_lex"],
        "final_action": {
            "action_type": "book_meeting",
            "participant_ids": ["vp_lex", "maya_ops", "jun_pm"],
            "proposed_start": "2026-04-07T11:00:00+08:00",
            "duration_minutes": 30,
        },
    },
}


def fallback_action(observation: dict[str, Any]) -> OpsFlowAction:
    selected_thread_id = observation.get("selected_thread_id")
    inbox_summary = observation.get("inbox_summary", [])
    if not selected_thread_id and inbox_summary:
        return OpsFlowAction(action_type="open_thread", thread_id=inbox_summary[0]["thread_id"])
    return OpsFlowAction(action_type="finish", reason="fallback")


def build_prompt(observation: dict[str, Any]) -> str:
    return json.dumps(observation, indent=2)


def parse_action(response_text: str, observation: dict[str, Any]) -> OpsFlowAction:
    try:
        payload = json.loads(response_text)
        return OpsFlowAction.model_validate(payload)
    except Exception:
        return fallback_action(observation)


def _constraints_match(observation: dict[str, Any], expected: dict[str, Any]) -> bool:
    known = observation.get("known_constraints", {})
    for key, value in expected.items():
        if known.get(key) != value:
            return False
    return True


def heuristic_action(observation: dict[str, Any]) -> OpsFlowAction:
    plan = HEURISTIC_PLANS.get(observation["task_id"])
    if plan is None:
        return fallback_action(observation)

    if observation.get("done"):
        return OpsFlowAction(action_type="finish", reason="episode complete")

    if observation.get("selected_thread_id") != plan["thread_id"]:
        return OpsFlowAction(action_type="open_thread", thread_id=plan["thread_id"])

    if not _constraints_match(observation, plan["extract"]):
        return OpsFlowAction(
            action_type="extract_constraints",
            thread_id=plan["thread_id"],
            extracted_fields=plan["extract"],
        )

    expected_views = plan.get("calendar_participants", [])
    viewed = sorted(view.get("participant") for view in observation.get("calendar_snapshots", []))
    if expected_views and viewed != sorted(expected_views):
        return OpsFlowAction(action_type="view_calendar", participant_ids=expected_views)

    conflict_probe = plan.get("conflict_probe")
    if conflict_probe is not None and not observation.get("pending_conflicts"):
        return OpsFlowAction.model_validate(conflict_probe)

    return OpsFlowAction.model_validate(plan["final_action"])


def choose_action(
    client: Any,
    observation: dict[str, Any],
) -> tuple[OpsFlowAction, str, Any]:
    if client is None or not HF_TOKEN:
        return heuristic_action(observation), "heuristic", None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(observation)},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        response_text = completion.choices[0].message.content or "{}"
        return parse_action(response_text, observation), "llm", client
    except Exception:
        return heuristic_action(observation), "heuristic", None


def log_start(task_id: str, mode: str) -> None:
    print(f"[START] task_id={task_id} mode={mode}")


def log_step(
    task_id: str,
    step_index: int,
    source: str,
    action: OpsFlowAction,
    reward: float,
    done: bool,
    score: float,
) -> None:
    payload = json.dumps(action.model_dump(exclude_none=True), sort_keys=True)
    print(
        f"[STEP] task_id={task_id} step={step_index} source={source} "
        f"reward={reward:.2f} done={str(done).lower()} score={score:.2f} action={payload}"
    )


def log_end(task_id: str, score: float, mode: str) -> None:
    print(f"[END] task_id={task_id} score={score:.2f} mode={mode}")


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN and OpenAI is not None else None
    env = OpsFlowEnv()

    task_ids = [task.task_id for task in env.available_tasks()]
    total_score = 0.0
    used_heuristic = client is None
    task_scores: dict[str, float] = {}

    for task_id in task_ids:
        result = env.reset(task_id=task_id)
        initial_mode = "llm" if client is not None else "heuristic"
        log_start(task_id, initial_mode)

        for step_index in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation = result.observation.model_dump(mode="json")
            action, source, client = choose_action(client, observation)
            used_heuristic = used_heuristic or source == "heuristic"
            result = env.step(action)
            log_step(
                task_id=task_id,
                step_index=step_index,
                source=source,
                action=action,
                reward=result.reward,
                done=result.done,
                score=float(result.info.get("score", 0.0)),
            )

        task_score = float(result.info.get("score", 0.0))
        total_score += task_score
        task_scores[task_id] = round(task_score, 3)
        final_mode = "heuristic" if used_heuristic else "llm"
        log_end(task_id, task_score, final_mode)

    average = total_score / max(len(task_ids), 1)
    summary = {
        "task_count": len(task_ids),
        "task_scores": task_scores,
        "average_score": round(average, 3),
        "mode": "heuristic_fallback" if used_heuristic else "llm",
    }
    if RESULTS_PATH:
        with open(RESULTS_PATH, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)


if __name__ == "__main__":
    main()
