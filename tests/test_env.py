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
            OpsFlowAction(action_type="classify_thread", category="meeting_request"),
            OpsFlowAction(action_type="set_priority", priority="high"),
            OpsFlowAction(
                action_type="extract_constraints",
                extracted_fields={
                    "request_type": "schedule_meeting",
                    "attendees": ["riya", "aman"],
                    "duration_minutes": 30,
                    "preferred_dates": ["2026-04-02"],
                    "preferred_time_ranges": ["afternoon"],
                    "timezone": "Asia/Calcutta",
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

    def test_medium_requires_clarification(self) -> None:
        self.env.reset(task_id="medium_missing_info_with_conflict")
        self.env.step(OpsFlowAction(action_type="open_thread", thread_id="t_medium_1"))
        self.env.step(
            OpsFlowAction(
                action_type="extract_constraints",
                extracted_fields={
                    "request_type": "schedule_meeting",
                    "attendees": ["priya", "karan"],
                    "preferred_dates": ["2026-04-04"],
                    "preferred_time_ranges": ["morning"],
                    "timezone": "Asia/Calcutta",
                    "missing_fields": ["duration_minutes"],
                },
            )
        )
        result = self.env.step(OpsFlowAction(action_type="ask_for_missing_info", reason="duration missing"))
        self.assertTrue(result.done)
        self.assertEqual(result.info["final_resolution"], "ask_for_missing_info")


if __name__ == "__main__":
    unittest.main()
