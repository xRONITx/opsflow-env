from __future__ import annotations

from datetime import datetime, timedelta

from opsflow_env.models import CalendarEvent, ConflictView, TaskDefinition


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def end_time(start_iso: str, duration_minutes: int) -> datetime:
    return parse_dt(start_iso) + timedelta(minutes=duration_minutes)


def detect_conflicts(task: TaskDefinition, participants: list[str], start_iso: str, duration_minutes: int) -> list[ConflictView]:
    proposed_end = end_time(start_iso, duration_minutes)
    conflicts: list[ConflictView] = []

    for participant in participants:
        for event in task.calendars.get(participant, []):
            event_start = parse_dt(event.start)
            event_end = parse_dt(event.end)
            if parse_dt(start_iso) < event_end and proposed_end > event_start:
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


def is_reschedule_allowed(task: TaskDefinition, event_id: str) -> bool:
    event = find_event(task, event_id)
    if event is None:
        return False
    if event.protected:
        return False
    disallowed = set(task.policies.get("non_movable_priorities", []))
    return event.priority not in disallowed
