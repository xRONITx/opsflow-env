def invalid_action_penalty() -> float:
    return -0.05


def loop_penalty() -> float:
    return -0.03


def reward_for_correct_progress(kind: str) -> float:
    table = {
        "open_thread": 0.10,
        "classify_thread": 0.10,
        "set_priority": 0.05,
        "extract_constraints": 0.15,
        "view_calendar": 0.10,
        "detect_conflict": 0.10,
        "final_resolution": 0.25,
        "archive_thread": 0.05,
    }
    return table.get(kind, 0.0)
