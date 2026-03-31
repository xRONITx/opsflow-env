from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from opsflow_env.env import OpsFlowEnv
from opsflow_env.models import OpsFlowAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
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


def main() -> None:
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME must be set before running inference.py")
    if not API_KEY:
        raise RuntimeError("HF_TOKEN or API_KEY must be set before running inference.py")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = OpsFlowEnv()

    task_ids = [task.task_id for task in env.available_tasks()]
    total_score = 0.0

    for task_id in task_ids:
        result = env.reset(task_id=task_id)
        print(f"\n=== Running task: {task_id} ===")

        for _ in range(MAX_STEPS):
            if result.done:
                break

            observation = result.observation.model_dump(mode="json")
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
            action = parse_action(response_text, observation)
            print(f"Action -> {action.model_dump(exclude_none=True)}")
            result = env.step(action)
            print(
                f"Reward={result.reward:.2f} Done={result.done} "
                f"Score={result.info.get('score', 0.0):.2f}"
            )

        task_score = float(result.info.get("score", 0.0))
        total_score += task_score
        print(f"Final score for {task_id}: {task_score:.2f}")

    average = total_score / max(len(task_ids), 1)
    summary = {
        "task_count": len(task_ids),
        "average_score": round(average, 3),
    }
    print(f"\nAverage score across {len(task_ids)} tasks: {average:.2f}")
    if RESULTS_PATH:
        with open(RESULTS_PATH, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)
        print(f"Wrote summary to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
