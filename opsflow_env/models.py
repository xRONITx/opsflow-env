from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "open_thread",
    "classify_thread",
    "set_priority",
    "extract_constraints",
    "view_calendar",
    "ask_for_missing_info",
    "propose_slot",
    "book_meeting",
    "reschedule_meeting",
    "decline_request",
    "escalate_request",
    "archive_thread",
    "finish",
]


class EmailMessage(BaseModel):
    sender: str
    sent_at: str
    body: str


class EmailThread(BaseModel):
    thread_id: str
    subject: str
    sender: str
    received_at: str
    category: str
    suggested_priority: str
    actionable: bool = True
    messages: list[EmailMessage]


class InboxItemSummary(BaseModel):
    thread_id: str
    subject: str
    sender: str
    received_at: str
    preview: Optional[str] = None
    is_opened: bool = False
    is_archived: bool = False


class CalendarEvent(BaseModel):
    event_id: str
    title: str
    start: str
    end: str
    priority: str
    protected: bool = False
    attendees: list[str] = Field(default_factory=list)


class CalendarView(BaseModel):
    participant: str
    timezone: str
    events: list[CalendarEvent]


class ExtractedConstraints(BaseModel):
    request_type: Optional[str] = None
    attendees: list[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = None
    preferred_dates: list[str] = Field(default_factory=list)
    preferred_time_ranges: list[str] = Field(default_factory=list)
    timezone: Optional[str] = None
    priority: Optional[str] = None
    deadline: Optional[str] = None
    missing_fields: list[str] = Field(default_factory=list)


class ConflictView(BaseModel):
    participant: str
    requested_slot: str
    conflicting_event_id: str
    reason: str


class TaskDefinition(BaseModel):
    task_id: str
    difficulty: str
    objective: str
    max_steps: int
    ideal_steps: int = 6
    inbox_threads: list[EmailThread]
    participants: dict[str, str]
    calendars: dict[str, list[CalendarEvent]]
    gold_constraints: dict[str, Any]
    expected_outcome: dict[str, Any]
    policies: dict[str, Any]
    follow_up_reply: Optional[EmailMessage] = None


class OpsFlowAction(BaseModel):
    action_type: ActionType
    thread_id: Optional[str] = None
    participant_ids: list[str] = Field(default_factory=list)
    category: Optional[str] = None
    priority: Optional[str] = None
    proposed_start: Optional[str] = None
    duration_minutes: Optional[int] = None
    target_event_id: Optional[str] = None
    reason: Optional[str] = None
    extracted_fields: dict[str, Any] = Field(default_factory=dict)


class EmailThreadView(BaseModel):
    thread_id: str
    subject: str
    messages: list[EmailMessage]


class OpsFlowObservation(BaseModel):
    task_id: str
    task_objective: str
    step_count: int
    max_steps: int
    inbox_summary: list[InboxItemSummary]
    participant_directory: dict[str, str] = Field(default_factory=dict)
    policy_hints: list[str] = Field(default_factory=list)
    selected_thread_id: Optional[str] = None
    selected_thread: Optional[EmailThreadView] = None
    known_constraints: ExtractedConstraints
    calendar_snapshots: list[CalendarView] = Field(default_factory=list)
    pending_conflicts: list[ConflictView] = Field(default_factory=list)
    last_action_result: Optional[str] = None
    last_action_error: bool = False
    done: bool = False


class OpsFlowState(BaseModel):
    task_id: str
    difficulty: str
    objective: str
    step_count: int
    max_steps: int
    selected_thread_id: Optional[str] = None
    opened_threads: list[str] = Field(default_factory=list)
    archived_threads: list[str] = Field(default_factory=list)
    viewed_calendars: list[str] = Field(default_factory=list)
    classified_threads: dict[str, str] = Field(default_factory=dict)
    thread_priorities: dict[str, str] = Field(default_factory=dict)
    known_constraints: ExtractedConstraints = Field(default_factory=ExtractedConstraints)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    pending_conflicts: list[ConflictView] = Field(default_factory=list)
    booked_meeting: Optional[dict[str, Any]] = None
    rescheduled_event: Optional[dict[str, Any]] = None
    final_resolution: Optional[str] = None
    clarification_requested: bool = False
    clarification_response_delivered: bool = False
    escalated: bool = False
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    done: bool = False
    last_action_result: Optional[str] = None
    last_action_error: bool = False


class StepResult(BaseModel):
    observation: OpsFlowObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
