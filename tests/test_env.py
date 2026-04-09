from __future__ import annotations

import unittest

from opsflow_env.env import OpsFlowEnv
from opsflow_env.models import OpsFlowAction


class OpsFlowEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = OpsFlowEnv()

    def test_reset_returns_clean_easy_state(self) -> None:
        result = self.env.reset(task_id="easy_single_request_clear_slot")
        self.assertEqual(result.observation.task_id, "easy_single_request_clear_slot")
        self.assertEqual(result.observation.step_count, 0)
        self.assertFalse(result.done)

    def test_easy_gold_trajectory_scores_full_credit(self) -> None:
        result = self.env.reset(task_id="easy_single_request_clear_slot")
        actions = [
            OpsFlowAction(action_type="open_thread", thread_id="t_easy_1"),
            OpsFlowAction(action_type="classify_thread", thread_id="t_easy_1", category="meeting_request"),
            OpsFlowAction(action_type="set_priority", thread_id="t_easy_1", priority="high"),
            OpsFlowAction(
                action_type="extract_constraints",
                thread_id="t_easy_1",
                extracted_fields={
                    "request_type": "schedule_meeting",
                    "attendees": ["riya", "aman"],
                    "duration_minutes": 30,
                    "preferred_dates": ["2026-04-02"],
                    "preferred_time_ranges": ["afternoon"],
                    "timezone": "Asia/Calcutta",
                    "priority": "high",
                    "missing_fields": [],
                },
            ),
            OpsFlowAction(action_type="view_calendar", participant_ids=["riya", "aman"]),
            OpsFlowAction(
                action_type="book_meeting",
                proposed_start="2026-04-02T15:00:00+05:30",
                participant_ids=["riya", "aman"],
                duration_minutes=30,
            ),
        ]
        for action in actions:
            result = self.env.step(action)
        self.assertTrue(result.done)
        self.assertEqual(result.info["score"], 1.0)

    def test_medium_clarification_flow_books_after_follow_up(self) -> None:
        result = self.env.reset(task_id="medium_missing_info_with_conflict")
        actions = [
            OpsFlowAction(action_type="open_thread", thread_id="t_medium_1"),
            OpsFlowAction(action_type="classify_thread", thread_id="t_medium_1", category="meeting_request"),
            OpsFlowAction(action_type="set_priority", thread_id="t_medium_1", priority="medium"),
            OpsFlowAction(
                action_type="extract_constraints",
                thread_id="t_medium_1",
                extracted_fields={
                    "request_type": "schedule_meeting",
                    "attendees": ["priya", "karan"],
                    "duration_minutes": None,
                    "preferred_dates": ["2026-04-04"],
                    "preferred_time_ranges": ["10:00", "late_morning"],
                    "timezone": "Asia/Calcutta",
                    "priority": "medium",
                    "missing_fields": ["duration_minutes"],
                },
            ),
            OpsFlowAction(action_type="ask_for_missing_info", thread_id="t_medium_1", reason="duration missing"),
        ]
        for action in actions:
            result = self.env.step(action)
        self.assertFalse(result.done)
        self.assertIn("follow-up", result.observation.last_action_result.lower())

        result = self.env.step(
            OpsFlowAction(
                action_type="extract_constraints",
                thread_id="t_medium_1",
                extracted_fields={
                    "request_type": "schedule_meeting",
                    "attendees": ["priya", "karan"],
                    "duration_minutes": 30,
                    "preferred_dates": ["2026-04-04"],
                    "preferred_time_ranges": ["10:00", "late_morning", "11:30"],
                    "timezone": "Asia/Calcutta",
                    "priority": "medium",
                    "missing_fields": [],
                },
            )
        )
        result = self.env.step(OpsFlowAction(action_type="view_calendar", participant_ids=["priya", "karan"]))
        result = self.env.step(
            OpsFlowAction(
                action_type="book_meeting",
                proposed_start="2026-04-04T11:30:00+05:30",
                participant_ids=["priya", "karan"],
                duration_minutes=30,
            )
        )
        self.assertTrue(result.done)
        self.assertEqual(result.info["score"], 1.0)

    def test_reschedule_task_scores_full_credit(self) -> None:
        result = self.env.reset(task_id="hard_reschedule_internal_sync")
        actions = [
            OpsFlowAction(action_type="open_thread", thread_id="t_reschedule_1"),
            OpsFlowAction(action_type="classify_thread", thread_id="t_reschedule_1", category="meeting_request"),
            OpsFlowAction(action_type="set_priority", thread_id="t_reschedule_1", priority="high"),
            OpsFlowAction(
                action_type="extract_constraints",
                thread_id="t_reschedule_1",
                extracted_fields={
                    "request_type": "schedule_meeting",
                    "attendees": ["anika_ops", "suraj_eng", "mehul_pm"],
                    "duration_minutes": 30,
                    "preferred_dates": ["2026-04-08"],
                    "preferred_time_ranges": ["10:00"],
                    "timezone": "Asia/Calcutta",
                    "priority": "high",
                    "deadline": "2026-04-08T10:30:00+05:30",
                    "missing_fields": [],
                },
            ),
            OpsFlowAction(action_type="view_calendar", participant_ids=["anika_ops", "suraj_eng", "mehul_pm"]),
            OpsFlowAction(
                action_type="reschedule_meeting",
                target_event_id="suraj_team_sync",
                proposed_start="2026-04-08T10:30:00+05:30",
            ),
            OpsFlowAction(
                action_type="book_meeting",
                proposed_start="2026-04-08T10:00:00+05:30",
                participant_ids=["anika_ops", "suraj_eng", "mehul_pm"],
                duration_minutes=30,
            ),
        ]
        for action in actions:
            result = self.env.step(action)
        self.assertTrue(result.done)
        self.assertEqual(result.info["score"], 1.0)

    def test_escalate_task_scores_full_credit(self) -> None:
        result = self.env.reset(task_id="hard_escalate_protected_conflict")
        actions = [
            OpsFlowAction(action_type="open_thread", thread_id="t_escalate_1"),
            OpsFlowAction(action_type="classify_thread", thread_id="t_escalate_1", category="meeting_request"),
            OpsFlowAction(action_type="set_priority", thread_id="t_escalate_1", priority="high"),
            OpsFlowAction(
                action_type="extract_constraints",
                thread_id="t_escalate_1",
                extracted_fields={
                    "request_type": "schedule_meeting",
                    "attendees": ["cto_ira", "lina_sales", "devon_ops"],
                    "duration_minutes": 30,
                    "preferred_dates": ["2026-04-09"],
                    "preferred_time_ranges": ["before_noon_singapore"],
                    "timezone": "Asia/Singapore",
                    "priority": "high",
                    "deadline": "2026-04-09T11:00:00+08:00",
                    "missing_fields": [],
                },
            ),
            OpsFlowAction(action_type="view_calendar", participant_ids=["cto_ira", "lina_sales", "devon_ops"]),
            OpsFlowAction(action_type="escalate_request", reason="Protected conflicts block every viable slot."),
        ]
        for action in actions:
            result = self.env.step(action)
        self.assertTrue(result.done)
        self.assertEqual(result.info["score"], 1.0)


if __name__ == "__main__":
    unittest.main()

