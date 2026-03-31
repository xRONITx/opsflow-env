from __future__ import annotations

from opsflow_env.models import OpsFlowState, TaskDefinition


def _score_easy(task: TaskDefinition, state: OpsFlowState) -> float:
    score = 0.0
    expected = task.expected_outcome
    constraints = task.gold_constraints

    if expected["thread_id"] in state.opened_threads:
        score += 0.2
    if sorted(state.known_constraints.attendees) == sorted(constraints["attendees"]):
        score += 0.2
    if state.known_constraints.duration_minutes == constraints["duration_minutes"]:
        score += 0.2
    if state.booked_meeting and state.booked_meeting.get("start") == expected["proposed_start"]:
        score += 0.25
    if state.final_resolution == "book_meeting" and not state.pending_conflicts:
        score += 0.15

    return round(min(score, 1.0), 3)


def _score_medium(task: TaskDefinition, state: OpsFlowState) -> float:
    score = 0.0
    expected = task.expected_outcome

    if expected["thread_id"] in state.opened_threads:
        score += 0.2
    if "duration_minutes" in state.known_constraints.missing_fields:
        score += 0.2
    if state.pending_conflicts:
        score += 0.2
    if state.clarification_requested:
        score += 0.25
    if state.final_resolution == "ask_for_missing_info" and state.booked_meeting is None:
        score += 0.15

    return round(min(score, 1.0), 3)


def _score_hard(task: TaskDefinition, state: OpsFlowState) -> float:
    score = 0.0
    expected = task.expected_outcome
    constraints = task.gold_constraints

    if expected["thread_id"] in state.opened_threads:
        score += 0.15
    if sorted(state.viewed_calendars) == sorted(expected["required_calendar_views"]):
        score += 0.15
    if sorted(state.known_constraints.attendees) == sorted(constraints["attendees"]):
        score += 0.15
    if state.known_constraints.timezone == constraints["timezone"]:
        score += 0.10
    if state.booked_meeting and state.booked_meeting.get("start") == expected["proposed_start"]:
        score += 0.25
    if state.final_resolution == "book_meeting":
        score += 0.10
    if state.booked_meeting and state.booked_meeting.get("conflicts") == []:
        score += 0.10

    return round(min(score, 1.0), 3)


def grade_task(task: TaskDefinition, state: OpsFlowState) -> float:
    if task.difficulty == "easy":
        return _score_easy(task, state)
    if task.difficulty == "medium":
        return _score_medium(task, state)
    return _score_hard(task, state)
