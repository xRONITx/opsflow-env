from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from opsflow_env.graders import grade_task_breakdown
from opsflow_env.models import (
    CalendarView,
    EmailThreadView,
    ExtractedConstraints,
    InboxItemSummary,
    OpsFlowAction,
    OpsFlowObservation,
    OpsFlowState,
    StepResult,
    TaskDefinition,
)
from opsflow_env.rewards import invalid_action_penalty, loop_penalty, reward_for_correct_progress
from opsflow_env.rules import detect_conflicts, find_event, is_reschedule_allowed, move_event, slot_respects_business_hours
from opsflow_env.tasks import load_all_tasks


class OpsFlowEnv:
    def __init__(self) -> None:
        self._tasks = {task.task_id: task for task in load_all_tasks()}
        self._task_order = list(self._tasks.keys())
        self._current_task: TaskDefinition | None = None
        self._state: OpsFlowState | None = None

    def available_tasks(self) -> list[TaskDefinition]:
        return [deepcopy(self._tasks[task_id]) for task_id in self._task_order]

    def reset(self, task_id: str | None = None) -> StepResult:
        chosen = task_id or self._task_order[0]
        self._current_task = deepcopy(self._tasks[chosen])
        self._state = OpsFlowState(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            objective=self._current_task.objective,
            step_count=0,
            max_steps=self._current_task.max_steps,
        )
        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"task_id": chosen, "score": 0.0, "score_breakdown": {}},
        )

    def state(self) -> OpsFlowState:
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling state().")
        return deepcopy(self._state)

    def step(self, action: OpsFlowAction) -> StepResult:
        if self._current_task is None or self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        state = self._state
        task = self._current_task

        if state.done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={
                    "message": "Episode already completed.",
                    "score": state.score,
                    "score_breakdown": state.score_breakdown,
                },
            )

        state.step_count += 1
        state.last_action_error = False
        state.last_action_result = None
        reward = 0.0

        action_signature = action.model_dump(exclude_none=True)
        if action_signature in state.action_history[-2:]:
            reward += loop_penalty()

        handler = getattr(self, f"_handle_{action.action_type}", None)
        if handler is None:
            reward += invalid_action_penalty()
            state.last_action_error = True
            state.last_action_result = f"Unsupported action_type={action.action_type}"
        else:
            reward += handler(action)

        state.action_history.append(action_signature)

        if state.step_count >= state.max_steps and not state.done:
            state.done = True
            state.last_action_result = (state.last_action_result or "") + " Max steps reached."

        if state.done:
            state.score_breakdown = grade_task_breakdown(task, state)
            state.score = state.score_breakdown["total"]

        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=round(reward, 3),
            done=state.done,
            info={
                "score": state.score,
                "score_breakdown": state.score_breakdown,
                "final_resolution": state.final_resolution,
                "pending_conflicts": [conflict.model_dump() for conflict in state.pending_conflicts],
            },
        )

    def _target_thread(self) -> str:
        if self._current_task is None:
            raise RuntimeError("No active task.")
        return str(self._current_task.expected_outcome["thread_id"])

    def _thread_by_id(self, thread_id: str):
        if self._current_task is None:
            raise RuntimeError("No active task.")
        for thread in self._current_task.inbox_threads:
            if thread.thread_id == thread_id:
                return thread
        return None

    def _require_selected_thread(self, thread_id: str | None):
        selected = thread_id or self._state.selected_thread_id  # type: ignore[union-attr]
        if selected is None:
            self._state.last_action_error = True  # type: ignore[union-attr]
            self._state.last_action_result = "No thread selected."  # type: ignore[union-attr]
            return None
        thread = self._thread_by_id(selected)
        if thread is None:
            self._state.last_action_error = True  # type: ignore[union-attr]
            self._state.last_action_result = f"Unknown thread_id={selected}"  # type: ignore[union-attr]
            return None
        return thread

    def _handle_open_thread(self, action: OpsFlowAction) -> float:
        state = self._state
        assert state is not None
        if not action.thread_id:
            state.last_action_error = True
            state.last_action_result = "open_thread requires thread_id."
            return invalid_action_penalty()

        if self._thread_by_id(action.thread_id) is None:
            state.last_action_error = True
            state.last_action_result = f"Unknown thread_id={action.thread_id}"
            return invalid_action_penalty()

        state.selected_thread_id = action.thread_id
        if action.thread_id not in state.opened_threads:
            state.opened_threads.append(action.thread_id)
        state.last_action_result = f"Opened thread {action.thread_id}."
        return reward_for_correct_progress("open_thread") if action.thread_id == self._target_thread() else 0.0

    def _handle_classify_thread(self, action: OpsFlowAction) -> float:
        state = self._state
        assert state is not None
        thread = self._require_selected_thread(action.thread_id)
        if thread is None or not action.category:
            return invalid_action_penalty()

        state.classified_threads[thread.thread_id] = action.category
        state.last_action_result = f"Classified {thread.thread_id} as {action.category}."
        return reward_for_correct_progress("classify_thread") if action.category == thread.category else 0.0

    def _handle_set_priority(self, action: OpsFlowAction) -> float:
        state = self._state
        assert state is not None
        thread = self._require_selected_thread(action.thread_id)
        if thread is None or not action.priority:
            return invalid_action_penalty()

        state.thread_priorities[thread.thread_id] = action.priority
        state.known_constraints.priority = action.priority
        state.last_action_result = f"Set priority for {thread.thread_id} to {action.priority}."
        return reward_for_correct_progress("set_priority") if action.priority == thread.suggested_priority else 0.0

    def _handle_extract_constraints(self, action: OpsFlowAction) -> float:
        state = self._state
        task = self._current_task
        assert state is not None and task is not None
        thread = self._require_selected_thread(action.thread_id)
        if thread is None or not action.extracted_fields:
            state.last_action_error = True
            state.last_action_result = "extract_constraints requires extracted_fields."
            return invalid_action_penalty()

        extracted = action.extracted_fields
        state.known_constraints = ExtractedConstraints(
            request_type=extracted.get("request_type"),
            attendees=list(extracted.get("attendees", [])),
            duration_minutes=extracted.get("duration_minutes"),
            preferred_dates=list(extracted.get("preferred_dates", [])),
            preferred_time_ranges=list(extracted.get("preferred_time_ranges", [])),
            timezone=extracted.get("timezone"),
            priority=extracted.get("priority") or state.thread_priorities.get(thread.thread_id) or state.known_constraints.priority,
            deadline=extracted.get("deadline"),
            missing_fields=list(extracted.get("missing_fields", [])),
        )

        gold = task.gold_constraints
        matched = 0
        if sorted(state.known_constraints.attendees) == sorted(gold.get("attendees", [])):
            matched += 1
        if state.known_constraints.duration_minutes == gold.get("duration_minutes"):
            matched += 1
        if state.known_constraints.timezone == gold.get("timezone"):
            matched += 1
        if sorted(state.known_constraints.preferred_dates) == sorted(gold.get("preferred_dates", [])):
            matched += 1
        if sorted(state.known_constraints.missing_fields) == sorted(gold.get("missing_fields", [])):
            matched += 1

        state.last_action_result = f"Extracted constraints for {thread.thread_id}."
        return reward_for_correct_progress("extract_constraints") + round(0.05 * matched / 5, 3)

    def _handle_view_calendar(self, action: OpsFlowAction) -> float:
        state = self._state
        task = self._current_task
        assert state is not None and task is not None
        if not action.participant_ids:
            state.last_action_error = True
            state.last_action_result = "view_calendar requires participant_ids."
            return invalid_action_penalty()

        unknown = [participant for participant in action.participant_ids if participant not in task.participants]
        if unknown:
            state.last_action_error = True
            state.last_action_result = f"Unknown participants: {', '.join(unknown)}."
            return invalid_action_penalty()

        for participant in action.participant_ids:
            if participant not in state.viewed_calendars:
                state.viewed_calendars.append(participant)
        state.last_action_result = f"Viewed calendars for {', '.join(action.participant_ids)}."
        return reward_for_correct_progress("view_calendar")

    def _handle_ask_for_missing_info(self, action: OpsFlowAction) -> float:
        state = self._state
        task = self._current_task
        assert state is not None and task is not None

        if not state.known_constraints.missing_fields:
            state.last_action_error = True
            state.last_action_result = "No missing fields detected."
            return invalid_action_penalty()

        if task.follow_up_reply is not None and not state.clarification_response_delivered:
            thread = self._require_selected_thread(action.thread_id)
            if thread is None:
                return invalid_action_penalty()
            thread.messages.append(deepcopy(task.follow_up_reply))
            state.clarification_requested = True
            state.clarification_response_delivered = True
            state.last_action_result = "Requested clarification and received a follow-up reply."
            return reward_for_correct_progress("clarification_reply")

        state.clarification_requested = True
        state.final_resolution = "ask_for_missing_info"
        state.done = True
        state.last_action_result = "Requested clarification from sender."
        return reward_for_correct_progress("final_resolution")

    def _handle_propose_slot(self, action: OpsFlowAction) -> float:
        state = self._state
        task = self._current_task
        assert state is not None and task is not None
        participants = action.participant_ids or state.known_constraints.attendees
        duration = action.duration_minutes or state.known_constraints.duration_minutes
        if not action.proposed_start or not participants or not duration:
            state.last_action_error = True
            state.last_action_result = "propose_slot requires proposed_start, participants, and duration."
            return invalid_action_penalty()

        if not slot_respects_business_hours(task, participants, action.proposed_start, duration):
            state.last_action_error = True
            state.last_action_result = "Proposed slot violates business hours."
            return -0.10

        conflicts = detect_conflicts(task, participants, action.proposed_start, duration)
        state.pending_conflicts = conflicts
        if conflicts:
            state.last_action_result = "Proposed slot has conflicts."
            return reward_for_correct_progress("detect_conflict")
        state.last_action_result = "Proposed slot is valid."
        return 0.05

    def _handle_book_meeting(self, action: OpsFlowAction) -> float:
        state = self._state
        task = self._current_task
        assert state is not None and task is not None
        participants = action.participant_ids or state.known_constraints.attendees
        duration = action.duration_minutes or state.known_constraints.duration_minutes
        if not action.proposed_start or not participants or not duration:
            state.last_action_error = True
            state.last_action_result = "book_meeting requires start, participants, and duration."
            return invalid_action_penalty()

        if state.known_constraints.missing_fields:
            state.last_action_error = True
            state.last_action_result = "Cannot book while required fields are still missing."
            return invalid_action_penalty()

        if not slot_respects_business_hours(task, participants, action.proposed_start, duration):
            state.last_action_error = True
            state.last_action_result = "Cannot book outside business hours."
            return -0.15

        conflicts = detect_conflicts(task, participants, action.proposed_start, duration)
        state.pending_conflicts = conflicts
        if conflicts:
            state.last_action_error = True
            state.last_action_result = "Cannot book conflicting meeting."
            return -0.15

        state.booked_meeting = {
            "start": action.proposed_start,
            "duration_minutes": duration,
            "participants": participants,
            "conflicts": [],
        }
        state.final_resolution = "book_meeting"
        state.done = True
        state.last_action_result = "Booked policy-compliant meeting."
        return reward_for_correct_progress("final_resolution")

    def _handle_reschedule_meeting(self, action: OpsFlowAction) -> float:
        state = self._state
        task = self._current_task
        assert state is not None and task is not None
        if not action.target_event_id or not action.proposed_start:
            state.last_action_error = True
            state.last_action_result = "reschedule_meeting requires target_event_id and proposed_start."
            return invalid_action_penalty()

        if not is_reschedule_allowed(task, action.target_event_id):
            state.last_action_error = True
            state.last_action_result = "Target event is protected and cannot be rescheduled."
            return -0.20

        event = find_event(task, action.target_event_id)
        if event is None:
            state.last_action_error = True
            state.last_action_result = f"Unknown target_event_id={action.target_event_id}."
            return invalid_action_penalty()

        original_start = event.start
        original_end = event.end
        duration = int((datetime.fromisoformat(original_end) - datetime.fromisoformat(original_start)).total_seconds() // 60)
        event_participants = event.attendees or [participant for participant, events in task.calendars.items() if any(item.event_id == event.event_id for item in events)]

        if not slot_respects_business_hours(task, event_participants, action.proposed_start, duration):
            state.last_action_error = True
            state.last_action_result = "Rescheduled slot violates business hours."
            return invalid_action_penalty()

        move_event(task, action.target_event_id, action.proposed_start)
        conflicts = [
            conflict
            for conflict in detect_conflicts(task, event_participants, action.proposed_start, duration)
            if conflict.conflicting_event_id != action.target_event_id
        ]
        if conflicts:
            event.start = original_start
            event.end = original_end
            state.pending_conflicts = conflicts
            state.last_action_error = True
            state.last_action_result = "Rescheduled event still conflicts with another meeting."
            return invalid_action_penalty()

        state.pending_conflicts = []
        state.rescheduled_event = {
            "target_event_id": action.target_event_id,
            "new_start": action.proposed_start,
        }
        state.last_action_result = f"Rescheduled {action.target_event_id}."
        return reward_for_correct_progress("reschedule_meeting")

    def _handle_decline_request(self, action: OpsFlowAction) -> float:
        state = self._state
        assert state is not None
        state.final_resolution = "decline_request"
        state.done = True
        state.last_action_result = "Declined request."
        return reward_for_correct_progress("final_resolution")

    def _handle_escalate_request(self, action: OpsFlowAction) -> float:
        state = self._state
        assert state is not None
        state.escalated = True
        state.final_resolution = "escalate_request"
        state.done = True
        state.last_action_result = "Escalated request."
        return reward_for_correct_progress("final_resolution")

    def _handle_archive_thread(self, action: OpsFlowAction) -> float:
        state = self._state
        thread = self._require_selected_thread(action.thread_id)
        if thread is None:
            return invalid_action_penalty()
        if thread.thread_id not in state.archived_threads:
            state.archived_threads.append(thread.thread_id)
        state.last_action_result = f"Archived {thread.thread_id}."
        return reward_for_correct_progress("archive_thread")

    def _handle_finish(self, action: OpsFlowAction) -> float:
        state = self._state
        assert state is not None
        state.done = True
        state.last_action_result = "Finished episode."
        return 0.0

    def _build_observation(self) -> OpsFlowObservation:
        if self._current_task is None or self._state is None:
            raise RuntimeError("Environment must be reset before observing state.")

        state = self._state
        task = self._current_task
        inbox_summary = [
            InboxItemSummary(
                thread_id=thread.thread_id,
                subject=thread.subject,
                sender=thread.sender,
                received_at=thread.received_at,
                preview=thread.messages[-1].body[:120] if thread.messages else None,
                is_opened=thread.thread_id in state.opened_threads,
                is_archived=thread.thread_id in state.archived_threads,
            )
            for thread in task.inbox_threads
        ]

        selected_thread = None
        if state.selected_thread_id:
            thread = self._thread_by_id(state.selected_thread_id)
            if thread:
                selected_thread = EmailThreadView(
                    thread_id=thread.thread_id,
                    subject=thread.subject,
                    messages=thread.messages,
                )

        calendar_snapshots = []
        for participant in state.viewed_calendars:
            calendar_snapshots.append(
                CalendarView(
                    participant=participant,
                    timezone=task.participants[participant],
                    events=task.calendars.get(participant, []),
                )
            )

        return OpsFlowObservation(
            task_id=task.task_id,
            task_objective=task.objective,
            step_count=state.step_count,
            max_steps=state.max_steps,
            inbox_summary=inbox_summary,
            participant_directory=task.participants,
            policy_hints=list(task.policies.get("policy_hints", [])),
            selected_thread_id=state.selected_thread_id,
            selected_thread=selected_thread,
            known_constraints=state.known_constraints,
            calendar_snapshots=calendar_snapshots,
            pending_conflicts=state.pending_conflicts,
            last_action_result=state.last_action_result,
            last_action_error=state.last_action_error,
            done=state.done,
        )


