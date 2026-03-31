from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_reset_endpoint(self) -> None:
        response = self.client.post("/reset", json={})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("observation", payload)
        self.assertEqual(payload["observation"]["task_id"], "easy_single_request_clear_slot")

    def test_state_endpoint_after_reset(self) -> None:
        self.client.post("/reset", json={"task_id": "medium_missing_info_with_conflict"})
        response = self.client.get("/state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task_id"], "medium_missing_info_with_conflict")


if __name__ == "__main__":
    unittest.main()
