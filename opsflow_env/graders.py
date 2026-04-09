from __future__ import annotations

from opsflow_env.models import OpsFlowState, TaskDefinition
from opsflow_env.rules import find_event, slot_respects_business_hours


def _target_thread_opened(task: TaskDefinition, state: OpsFlowState) -> float:
    return 0.15 if task.expected_outcome["thread_id"] in state.opened_threads else 0.0


def _classification_score(task: TaskDefinition, state: OpsFlowState) -> float:
    target_thread_id = task.expected_outcome["thread_id"]
    thread = next(thread for thread in task.inbox_threads if thread.thread_id == target_thread_id)
    return 0.10 if state.classified_threads.get(target_thread_id) == thread.category else 0.0


def _priority_score(task: TaskDefinition, state: OpsFlowState) -> float:
    target_thread_id = task.expected_outcome["thread_id"]
    thread = next(thread for thread in task.inbox_threads if thread.thread_id == target_thread_id)
    chosen_priority = state.thread_priorities.get(target_thread_id) or state.known_constraints.priority
    return 0.05 if chosen_priority == thread.suggested_priority else 0.0


def _constraint_score(task: TaskDefinition, state: OpsFlowState) -> float:
    expected = task.gold_constraints
    actual = state.known_constraints
    checks = [
        sorted(actual.attendees) == sorted(expected.get("attendees", [])),
        actual.duration_minutes == expected.get("duration_minutes"),
        actual.timezone == expected.get("timezone"),
        sorted(actual.preferred_dates) == sorted(expected.get("preferred_dates", [])),
        sorted(actual.missing_fields) == sorted(expected.get("missing_fields", [])),
    ]
    return round(0.20 * (sum(1 for item in checks if item) / len(checks)), 3)


def _calendar_score(task: TaskDefinition, state: OpsFlowState) -> float:
    required = task.expected_outcome.get("required_calendar_views") or task.gold_constraints.get("attendees", [])
    return 0.10 if sorted(state.viewed_calendars) == sorted(required) else 0.0


def _conflict_handling_score(task: TaskDefinition, state: OpsFlowState) -> float:
    expected = task.expected_outcome
    if task.follow_up_reply is not None:
        if state.clarification_requested and state.clarification_response_delivered:
            return 0.15
        return 0.0
    if expected.get("required_reschedule_event"):
        if (
            state.rescheduled_event
            and state.rescheduled_event.get("target_event_id") == expected["required_reschedule_event"]
            and state.rescheduled_event.get("new_start") == expected.get("required_reschedule_start")
        ):
            return 0.15
        return 0.0
    if expected.get("final_resolution") == "escalate_request":
        return 0.15 if state.escalated else 0.0
    if expected.get("final_resolution") == "decline_request":
        return 0.15 if state.final_resolution == "decline_request" else 0.0
    if state.final_resolution == "book_meeting" and not state.pending_conflicts:
        return 0.15
    return 0.0


def _resolution_score(task: TaskDefinition, state: OpsFlowState) -> float:
    expected = task.expected_outcome
    final_resolution = expected.get("final_resolution")
    if final_resolution != state.final_resolution:
        return 0.0
    if final_resolution == "book_meeting":
        if state.booked_meeting and state.booked_meeting.get("start") == expected.get("proposed_start"):
            return 0.20
        return 0.10
    return 0.20


def _policy_compliance_score(task: TaskDefinition, state: OpsFlowState) -> float:
    if state.final_resolution == "book_meeting":
        if state.booked_meeting is None:
            return 0.0
        if state.pending_conflicts:
            return 0.0
        if not slot_respects_business_hours(
            task,
            state.booked_meeting.get("participants", []),
            state.booked_meeting["start"],
            int(state.booked_meeting["duration_minutes"]),
        ):
            return 0.0
        return 0.10

    if state.rescheduled_event is not None:
        target_event = find_event(task, state.rescheduled_event["target_event_id"])
        if target_event is None or target_event.protected:
            return 0.0
    return 0.10 if not state.last_action_error else 0.0


def _efficiency_score(task: TaskDefinition, state: OpsFlowState) -> float:
    overrun = max(0, state.step_count - task.ideal_steps)
    return round(max(0.0, 0.05 - (overrun * 0.01)), 3)


def grade_task_breakdown(task: TaskDefinition, state: OpsFlowState) -> dict[str, float]:
    breakdown = {
        "thread_selection": _target_thread_opened(task, state),
        "classification": _classification_score(task, state),
        "priority": _priority_score(task, state),
        "constraint_extraction": _constraint_score(task, state),
        "calendar_review": _calendar_score(task, state),
        "conflict_handling": _conflict_handling_score(task, state),
        "resolution": _resolution_score(task, state),
        "policy_compliance": _policy_compliance_score(task, state),
        "efficiency": _efficiency_score(task, state),
    }
    breakdown["total"] = round(min(sum(breakdown.values()), 1.0), 3)
    return breakdown


def grade_task(task: TaskDefinition, state: OpsFlowState) -> float:
    return grade_task_breakdown(task, state)["total"]
