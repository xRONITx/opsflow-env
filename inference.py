from __future__ import annotations

import json
import os
import re
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

from opsflow_env.env import OpsFlowEnv
from opsflow_env.models import OpsFlowAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3-0324")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 14
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

MEETING_KEYWORDS = ("schedule", "sync", "meeting", "review", "call", "book")
NOISE_KEYWORDS = ("newsletter", "receipt", "restock", "travel", "pantry", "orientation")
PRIORITY_KEYWORDS = {
    "high": ("urgent", "high priority", "prioritize", "exec", "launch", "demo prep"),
    "low": ("no rush", "next week", "later this month", "when there is time", "casual"),
}
TIME_RANGE_TOKENS = {
    "afternoon": ["15:00", "15:30", "16:00", "16:30"],
    "morning": ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30"],
    "late_morning": ["11:00", "11:30"],
    "before_noon_singapore": ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30"],
}
BUSINESS_START = time.fromisoformat("08:30")
BUSINESS_END = time.fromisoformat("18:00")


def build_prompt(observation: dict[str, Any]) -> str:
    return json.dumps(observation, indent=2)


def _selected_thread(observation: dict[str, Any]) -> dict[str, Any] | None:
    return observation.get("selected_thread")


def _selected_summary(observation: dict[str, Any]) -> dict[str, Any] | None:
    selected_id = observation.get("selected_thread_id")
    if selected_id is None:
        return None
    for summary in observation.get("inbox_summary", []):
        if summary["thread_id"] == selected_id:
            return summary
    return None


def _thread_text(thread: dict[str, Any] | None) -> str:
    if not thread:
        return ""
    parts = [thread.get("subject", ""), thread.get("thread_id", "")]
    for message in thread.get("messages", []):
        parts.append(message.get("sender", ""))
        parts.append(message.get("body", ""))
    return " ".join(parts)


def _participant_aliases(participant_id: str) -> set[str]:
    parts = participant_id.lower().split("_")
    aliases = {participant_id.lower(), participant_id.lower().replace("_", " ")}
    if parts:
        aliases.add(parts[-1])
        aliases.add(parts[0])
    if len(parts) > 1:
        aliases.add(" ".join(parts))
        aliases.add(parts[-1])
    return {alias for alias in aliases if alias and alias not in {"ops", "pm", "eng", "sales"}}


def _meeting_score(summary: dict[str, Any]) -> int:
    text = f"{summary.get('subject', '')} {summary.get('preview', '')}".lower()
    score = 0
    if any(keyword in text for keyword in MEETING_KEYWORDS):
        score += 3
    if any(keyword in text for keyword in PRIORITY_KEYWORDS["high"]):
        score += 4
    if any(keyword in text for keyword in PRIORITY_KEYWORDS["low"]):
        score -= 2
    if any(keyword in text for keyword in NOISE_KEYWORDS):
        score -= 4
    return score


def choose_thread(observation: dict[str, Any]) -> str | None:
    candidates = [
        summary
        for summary in observation.get("inbox_summary", [])
        if not summary.get("is_archived")
    ]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: (_meeting_score(item), item.get("received_at", "")), reverse=True)
    return ranked[0]["thread_id"]


def infer_category(thread: dict[str, Any] | None) -> str:
    text = _thread_text(thread).lower()
    return "meeting_request" if any(keyword in text for keyword in MEETING_KEYWORDS) else "informational"


def infer_priority(thread: dict[str, Any] | None) -> str:
    text = _thread_text(thread).lower()
    if any(keyword in text for keyword in PRIORITY_KEYWORDS["high"]):
        return "high"
    if any(keyword in text for keyword in PRIORITY_KEYWORDS["low"]):
        return "low"
    return "medium"


def infer_attendees(thread: dict[str, Any] | None, participant_directory: dict[str, str]) -> list[str]:
    text = _thread_text(thread).lower()
    attendees: list[str] = []
    for participant_id in participant_directory:
        if any(alias in text for alias in _participant_aliases(participant_id)):
            attendees.append(participant_id)
    return attendees


def infer_duration_minutes(thread: dict[str, Any] | None) -> int | None:
    text = _thread_text(thread).lower()
    match = re.search(r"(\d+)\s*[- ]?(?:minute|min)", text)
    if match:
        return int(match.group(1))
    if "half hour" in text:
        return 30
    return None


def infer_dates(thread: dict[str, Any] | None, received_at: str | None) -> list[str]:
    text = _thread_text(thread)
    dates = re.findall(r"20\d{2}-\d{2}-\d{2}", text)
    if dates:
        return sorted(set(dates))
    if received_at and "tomorrow" in text.lower():
        return [(datetime.fromisoformat(received_at) + timedelta(days=1)).date().isoformat()]
    return []


def _normalize_clock(hour: int, minute: int, meridiem: str | None) -> str:
    if meridiem:
        meridiem = meridiem.lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    return f"{hour:02d}:{minute:02d}"


def infer_time_preferences(thread: dict[str, Any] | None) -> list[str]:
    text = _thread_text(thread).lower()
    preferences: list[str] = []

    for hour, minute, meridiem in re.findall(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text):
        preferences.append(_normalize_clock(int(hour), int(minute or 0), meridiem))

    for hour, minute in re.findall(r"\b(\d{1,2}):(\d{2})\b", text):
        token = f"{int(hour):02d}:{int(minute):02d}"
        if token not in preferences:
            preferences.append(token)

    if "late morning" in text:
        preferences.append("late_morning")
    elif "morning" in text:
        preferences.append("morning")
    if "afternoon" in text:
        preferences.append("afternoon")
    if "singapore time" in text and "before" in text:
        preferences.append("before_noon_singapore")

    unique_preferences: list[str] = []
    for preference in preferences:
        if preference not in unique_preferences:
            unique_preferences.append(preference)
    return unique_preferences


def infer_timezone(thread: dict[str, Any] | None, attendees: list[str], participant_directory: dict[str, str]) -> str | None:
    text = _thread_text(thread).lower()
    if "singapore" in text:
        return "Asia/Singapore"
    if "kolkata" in text or "calcutta" in text or "india" in text:
        return "Asia/Calcutta"
    attendee_timezones = {participant_directory.get(attendee) for attendee in attendees}
    attendee_timezones.discard(None)
    if len(attendee_timezones) == 1:
        return attendee_timezones.pop()
    return "Asia/Calcutta" if attendee_timezones else None


def infer_deadline(thread: dict[str, Any] | None, preferred_dates: list[str], timezone_name: str | None) -> str | None:
    text = _thread_text(thread).lower()
    if not preferred_dates or not timezone_name:
        return None
    match = re.search(r"before\s+(\d{1,2})(?::(\d{2}))?", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    date_value = datetime.fromisoformat(preferred_dates[0])
    return datetime.combine(date_value.date(), time(hour, minute), tzinfo=ZoneInfo(timezone_name)).isoformat()


def infer_constraints(observation: dict[str, Any]) -> dict[str, Any]:
    thread = _selected_thread(observation)
    summary = _selected_summary(observation)
    participant_directory = observation.get("participant_directory", {})
    attendees = infer_attendees(thread, participant_directory)
    duration_minutes = infer_duration_minutes(thread)
    preferred_dates = infer_dates(thread, summary.get("received_at") if summary else None)
    preferred_time_ranges = infer_time_preferences(thread)
    timezone_name = infer_timezone(thread, attendees, participant_directory)
    priority = infer_priority(thread)
    deadline = infer_deadline(thread, preferred_dates, timezone_name)
    missing_fields: list[str] = []
    if duration_minutes is None:
        missing_fields.append("duration_minutes")
    if not attendees:
        missing_fields.append("attendees")

    return {
        "request_type": "schedule_meeting" if infer_category(thread) == "meeting_request" else "informational",
        "attendees": attendees,
        "duration_minutes": duration_minutes,
        "preferred_dates": preferred_dates,
        "preferred_time_ranges": preferred_time_ranges,
        "timezone": timezone_name,
        "priority": priority,
        "deadline": deadline,
        "missing_fields": missing_fields,
    }


def constraints_need_update(known: dict[str, Any], inferred: dict[str, Any]) -> bool:
    keys = [
        "request_type",
        "attendees",
        "duration_minutes",
        "preferred_dates",
        "preferred_time_ranges",
        "timezone",
        "priority",
        "deadline",
        "missing_fields",
    ]
    for key in keys:
        known_value = known.get(key)
        inferred_value = inferred.get(key)
        if isinstance(known_value, list):
            known_value = sorted(known_value)
        if isinstance(inferred_value, list):
            inferred_value = sorted(inferred_value)
        if known_value != inferred_value:
            return True
    return False


def viewed_participants(observation: dict[str, Any]) -> set[str]:
    return {view["participant"] for view in observation.get("calendar_snapshots", [])}


def build_calendar_index(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {view["participant"]: view for view in observation.get("calendar_snapshots", [])}


def slot_conflicts(
    observation: dict[str, Any],
    participants: list[str],
    start_iso: str,
    duration_minutes: int,
    ignore_event_id: str | None = None,
) -> list[dict[str, Any]]:
    proposed_start = datetime.fromisoformat(start_iso)
    proposed_end = proposed_start + timedelta(minutes=duration_minutes)
    calendars = build_calendar_index(observation)
    conflicts: list[dict[str, Any]] = []

    for participant in participants:
        view = calendars.get(participant)
        if view is None:
            continue
        for event in view.get("events", []):
            if ignore_event_id and event["event_id"] == ignore_event_id:
                continue
            event_start = datetime.fromisoformat(event["start"])
            event_end = datetime.fromisoformat(event["end"])
            if proposed_start < event_end and proposed_end > event_start:
                conflicts.append(
                    {
                        "participant": participant,
                        "event_id": event["event_id"],
                        "priority": event.get("priority", "medium"),
                        "protected": bool(event.get("protected", False)),
                        "title": event.get("title", ""),
                    }
                )
    return conflicts


def slot_within_business_hours(
    observation: dict[str, Any],
    participants: list[str],
    start_iso: str,
    duration_minutes: int,
) -> bool:
    participant_directory = observation.get("participant_directory", {})
    start = datetime.fromisoformat(start_iso)
    end = start + timedelta(minutes=duration_minutes)
    for participant in participants:
        timezone_name = participant_directory.get(participant)
        if timezone_name is None:
            return False
        local_start = start.astimezone(ZoneInfo(timezone_name))
        local_end = end.astimezone(ZoneInfo(timezone_name))
        if local_start.timetz().replace(tzinfo=None) < BUSINESS_START:
            return False
        if local_end.timetz().replace(tzinfo=None) > BUSINESS_END:
            return False
    return True


def generate_candidate_slots(constraints: dict[str, Any]) -> list[str]:
    dates = constraints.get("preferred_dates") or []
    preferences = constraints.get("preferred_time_ranges") or ["morning"]
    deadline = constraints.get("deadline")
    duration = constraints.get("duration_minutes") or 30
    candidates: list[str] = []

    for date_value in dates:
        for preference in preferences:
            times = TIME_RANGE_TOKENS.get(preference, [preference])
            for time_token in times:
                if not re.fullmatch(r"\d{2}:\d{2}", time_token):
                    continue
                candidate = f"{date_value}T{time_token}:00"
                timezone_name = constraints.get("timezone") or "Asia/Calcutta"
                aware_candidate = datetime.fromisoformat(candidate).replace(tzinfo=ZoneInfo(timezone_name)).isoformat()
                if deadline:
                    if datetime.fromisoformat(aware_candidate) + timedelta(minutes=duration) > datetime.fromisoformat(deadline):
                        continue
                candidates.append(aware_candidate)

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def find_next_reschedule_slot(
    observation: dict[str, Any],
    participant: str,
    event_id: str,
    duration_minutes: int,
    preferred_date: str,
    earliest_start_iso: str,
) -> str | None:
    start = datetime.fromisoformat(earliest_start_iso)
    end_of_day = datetime.fromisoformat(f"{preferred_date}T17:30:00{start.strftime('%z')[:3]}:{start.strftime('%z')[3:]}")
    candidate = start
    while candidate <= end_of_day:
        candidate_iso = candidate.isoformat()
        conflicts = slot_conflicts(observation, [participant], candidate_iso, duration_minutes, ignore_event_id=event_id)
        if not conflicts and slot_within_business_hours(observation, [participant], candidate_iso, duration_minutes):
            return candidate_iso
        candidate += timedelta(minutes=30)
    return None


def choose_slot_action(observation: dict[str, Any], constraints: dict[str, Any]) -> OpsFlowAction:
    attendees = constraints.get("attendees") or []
    duration = constraints.get("duration_minutes")
    if not attendees or duration is None:
        return OpsFlowAction(action_type="finish", reason="insufficient constraints")

    candidate_slots = generate_candidate_slots(constraints)
    slot_evaluations: list[tuple[str, list[dict[str, Any]]]] = []
    for slot in candidate_slots:
        if not slot_within_business_hours(observation, attendees, slot, duration):
            continue
        conflicts = slot_conflicts(observation, attendees, slot, duration)
        slot_evaluations.append((slot, conflicts))
        if not conflicts:
            return OpsFlowAction(
                action_type="book_meeting",
                participant_ids=attendees,
                proposed_start=slot,
                duration_minutes=duration,
            )

    if slot_evaluations:
        preferred_slot, conflicts = slot_evaluations[0]
        if conflicts and all((not conflict["protected"]) and conflict["priority"] not in {"high", "critical"} for conflict in conflicts):
            conflict = conflicts[0]
            new_start = find_next_reschedule_slot(
                observation,
                participant=conflict["participant"],
                event_id=conflict["event_id"],
                duration_minutes=duration,
                preferred_date=constraints["preferred_dates"][0],
                earliest_start_iso=(datetime.fromisoformat(preferred_slot) + timedelta(minutes=duration)).isoformat(),
            )
            if new_start:
                return OpsFlowAction(
                    action_type="reschedule_meeting",
                    target_event_id=conflict["event_id"],
                    proposed_start=new_start,
                    reason="Free the requested urgent slot.",
                )
        if any(conflict["protected"] or conflict["priority"] in {"high", "critical"} for conflict in conflicts):
            return OpsFlowAction(action_type="escalate_request", reason="Protected conflicts block every viable slot.")

    if "escalate" in " ".join(observation.get("policy_hints", [])).lower() or constraints.get("deadline"):
        return OpsFlowAction(action_type="escalate_request", reason="No viable slot remains before the deadline.")

    return OpsFlowAction(action_type="finish", reason="no_safe_action")


def heuristic_action(observation: dict[str, Any], memory: dict[str, Any]) -> OpsFlowAction:
    thread = _selected_thread(observation)
    if thread is None:
        thread_id = choose_thread(observation)
        if thread_id is None:
            return OpsFlowAction(action_type="finish", reason="empty inbox")
        return OpsFlowAction(action_type="open_thread", thread_id=thread_id)

    thread_id = thread["thread_id"]
    inferred = infer_constraints(observation)

    classified_threads = memory.setdefault("classified_threads", set())
    prioritized_threads = memory.setdefault("prioritized_threads", set())
    clarification_requests = memory.setdefault("clarification_requests", set())

    if thread_id not in classified_threads:
        classified_threads.add(thread_id)
        return OpsFlowAction(action_type="classify_thread", thread_id=thread_id, category=infer_category(thread))

    if thread_id not in prioritized_threads:
        prioritized_threads.add(thread_id)
        return OpsFlowAction(action_type="set_priority", thread_id=thread_id, priority=infer_priority(thread))

    if constraints_need_update(observation.get("known_constraints", {}), inferred):
        return OpsFlowAction(action_type="extract_constraints", thread_id=thread_id, extracted_fields=inferred)

    if inferred.get("missing_fields") and thread_id not in clarification_requests:
        clarification_requests.add(thread_id)
        return OpsFlowAction(action_type="ask_for_missing_info", thread_id=thread_id, reason="Need required scheduling details.")

    attendees = observation.get("known_constraints", {}).get("attendees") or inferred.get("attendees") or []
    if attendees:
        unseen = [participant for participant in attendees if participant not in viewed_participants(observation)]
        if unseen:
            return OpsFlowAction(action_type="view_calendar", participant_ids=unseen)

    current_constraints = observation.get("known_constraints", {})
    effective_constraints = {
        "attendees": current_constraints.get("attendees") or inferred.get("attendees") or [],
        "duration_minutes": current_constraints.get("duration_minutes") or inferred.get("duration_minutes"),
        "preferred_dates": current_constraints.get("preferred_dates") or inferred.get("preferred_dates") or [],
        "preferred_time_ranges": current_constraints.get("preferred_time_ranges") or inferred.get("preferred_time_ranges") or ["morning"],
        "timezone": current_constraints.get("timezone") or inferred.get("timezone") or "Asia/Calcutta",
        "priority": current_constraints.get("priority") or inferred.get("priority"),
        "deadline": current_constraints.get("deadline") or inferred.get("deadline"),
        "missing_fields": current_constraints.get("missing_fields") or inferred.get("missing_fields") or [],
    }

    if effective_constraints["missing_fields"]:
        return OpsFlowAction(action_type="finish", reason="awaiting clarification")

    return choose_slot_action(observation, effective_constraints)


def parse_action(response_text: str, observation: dict[str, Any], memory: dict[str, Any]) -> OpsFlowAction:
    try:
        payload = json.loads(response_text)
        return OpsFlowAction.model_validate(payload)
    except Exception:
        return heuristic_action(observation, memory)


def choose_action(client: Any, observation: dict[str, Any], memory: dict[str, Any]) -> tuple[OpsFlowAction, str, Any]:
    if client is None or not HF_TOKEN:
        return heuristic_action(observation, memory), "heuristic", None

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
        return parse_action(response_text, observation, memory), "llm", client
    except Exception:
        return heuristic_action(observation, memory), "heuristic", None


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
    task_scores: dict[str, float] = {}
    used_heuristic_globally = client is None

    for task_id in task_ids:
        result = env.reset(task_id=task_id)
        memory: dict[str, Any] = {}
        initial_mode = "llm" if client is not None else "heuristic"
        log_start(task_id, initial_mode)

        for step_index in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation = result.observation.model_dump(mode="json")
            action, source, client = choose_action(client, observation, memory)
            used_heuristic_globally = used_heuristic_globally or source == "heuristic"
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
        final_mode = "heuristic" if used_heuristic_globally else "llm"
        log_end(task_id, task_score, final_mode)

    average = total_score / max(len(task_ids), 1)
    summary = {
        "task_count": len(task_ids),
        "task_scores": task_scores,
        "average_score": round(average, 3),
        "mode": "heuristic_fallback" if used_heuristic_globally else "llm",
    }
    if RESULTS_PATH:
        with open(RESULTS_PATH, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)


if __name__ == "__main__":
    main()

