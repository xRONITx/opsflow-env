from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from opsflow_env.models import CalendarEvent, ConflictView, TaskDefinition


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def end_time(start_iso: str, duration_minutes: int) -> datetime:
    return parse_dt(start_iso) + timedelta(minutes=duration_minutes)


def detect_conflicts(task: TaskDefinition, participants: list[str], start_iso: str, duration_minutes: int) -> list[ConflictView]:
    proposed_start = parse_dt(start_iso)
    proposed_end = end_time(start_iso, duration_minutes)
    conflicts: list[ConflictView] = []

    for participant in participants:
        for event in task.calendars.get(participant, []):
            event_start = parse_dt(event.start)
            event_end = parse_dt(event.end)
            if proposed_start < event_end and proposed_end > event_start:
                conflicts.append(
                    ConflictView(
                        participant=participant,
                        requested_slot=f"{start_iso}/{proposed_end.isoformat()}",
                        conflicting_event_id=event.event_id,
                        reason="overlaps existing event",
                    )
                )

    return conflicts


def find_event(task: TaskDefinition, event_id: str) -> CalendarEvent | None:
    for events in task.calendars.values():
        for event in events:
            if event.event_id == event_id:
                return event
    return None


def event_owner(task: TaskDefinition, event_id: str) -> str | None:
    for participant, events in task.calendars.items():
        for event in events:
            if event.event_id == event_id:
                return participant
    return None


def is_reschedule_allowed(task: TaskDefinition, event_id: str) -> bool:
    event = find_event(task, event_id)
    if event is None:
        return False
    if event.protected:
        return False
    disallowed = set(task.policies.get("non_movable_priorities", []))
    return event.priority not in disallowed


def slot_respects_business_hours(task: TaskDefinition, participants: list[str], start_iso: str, duration_minutes: int) -> bool:
    business_hours = task.policies.get("business_hours", {"start": "09:00", "end": "18:00"})
    window_start = time.fromisoformat(business_hours.get("start", "09:00"))
    window_end = time.fromisoformat(business_hours.get("end", "18:00"))
    proposed_start = parse_dt(start_iso)
    proposed_end = end_time(start_iso, duration_minutes)

    for participant in participants:
        timezone_name = task.participants.get(participant)
        if timezone_name is None:
            return False
        participant_zone = ZoneInfo(timezone_name)
        local_start = proposed_start.astimezone(participant_zone)
        local_end = proposed_end.astimezone(participant_zone)
        if local_start.timetz().replace(tzinfo=None) < window_start:
            return False
        if local_end.timetz().replace(tzinfo=None) > window_end:
            return False
    return True


def move_event(task: TaskDefinition, event_id: str, new_start_iso: str) -> CalendarEvent | None:
    event = find_event(task, event_id)
    if event is None:
        return None
    duration_minutes = int((parse_dt(event.end) - parse_dt(event.start)).total_seconds() // 60)
    event.start = new_start_iso
    event.end = end_time(new_start_iso, duration_minutes).isoformat()
    return event
