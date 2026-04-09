---
title: OpsFlowEnv
sdk: docker
app_port: 7860
short_description: OpenEnv benchmark for ops inbox triage and scheduling.
---

# OpsFlowEnv

OpsFlowEnv is an OpenEnv-style benchmark where an AI agent acts as an operations assistant. The agent triages inbox threads, extracts scheduling constraints from email requests, inspects participant calendars, resolves conflicts, and makes policy-compliant scheduling decisions.

## Why this environment matters

This environment models a real operations workflow rather than a toy task. Human coordinators, executive assistants, and program managers routinely:

- scan incoming email threads
- identify which requests are actionable
- determine urgency and scheduling intent
- inspect calendars
- resolve conflicts
- ask for clarification when needed
- book, reschedule, or escalate meetings

OpsFlowEnv turns that workflow into a deterministic benchmark with explicit step-by-step actions, multi-step clarification, rescheduling, escalation, and rubric-based grading.

## OpenEnv compliance

The project provides:

- typed `Action`, `Observation`, and state models with Pydantic
- `reset()`, `step()`, and `state()` APIs
- `openenv.yaml`
- an HTTP server exposing `/reset`, `/step`, and `/state`
- a working `Dockerfile`
- a root-level `inference.py` using the OpenAI client and the required environment variables

## Tasks

The benchmark includes five graded tasks that cover booking, clarification, rescheduling, and escalation.

### 1. `easy_single_request_clear_slot`

- One primary scheduling thread plus distractor inbox noise
- Complete meeting information
- One obvious valid slot
- Expected resolution: book the correct meeting

### 2. `medium_missing_info_with_conflict`

- Request is missing duration
- Requested window conflicts with another event
- Asking for clarification reveals a follow-up email reply
- Expected resolution: request missing information, update constraints, and then book the correct slot

### 3. `hard_priority_conflict_timezone`

- Urgent multi-party scheduling request
- Timezone-sensitive attendees
- Protected existing events cannot be moved automatically
- Expected resolution: inspect calendars and book the only policy-compliant slot

### 4. `hard_reschedule_internal_sync`

- Requested slot is blocked by a movable internal sync
- The agent must free the slot by rescheduling the lower-priority meeting
- Expected resolution: reschedule the internal sync and then book the requested review

### 5. `hard_escalate_protected_conflict`

- All realistic slots are blocked by protected or critical meetings
- Deadline pressure prevents simply deferring the task
- Expected resolution: escalate rather than forcing an invalid booking

## Action space

The agent acts using one structured action at a time:

- `open_thread`
- `classify_thread`
- `set_priority`
- `extract_constraints`
- `view_calendar`
- `ask_for_missing_info`
- `propose_slot`
- `book_meeting`
- `reschedule_meeting`
- `decline_request`
- `escalate_request`
- `archive_thread`
- `finish`

## Observation space

Each observation includes:

- current `task_id` and task objective
- current step count and max steps
- inbox summaries
- participant directory and policy hints
- currently selected thread content
- currently known extracted constraints
- calendar snapshots already requested by the agent
- unresolved conflicts
- previous action result and error state

## Reward design

The reward function includes partial progress:

- positive reward for opening the right thread
- positive reward for correct classification and priority handling
- positive reward for extracting correct scheduling constraints
- positive reward when a clarification reply is triggered correctly
- positive reward for inspecting relevant calendars
- positive reward for valid reschedules, safe bookings, or correct escalation
- penalties for invalid actions, policy violations, conflicting bookings, and loops

## Deterministic grading

Each task has a deterministic grader that returns a score from `0.0` to `1.0` and stores a rubric breakdown across:

- thread selection
- classification
- priority handling
- constraint extraction
- calendar review
- conflict handling
- final resolution
- policy compliance
- efficiency

## Local setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Inference script

The submission baseline script is `inference.py` in the repo root.

Remote LLM mode supports these environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Run it with:

```bash
python inference.py
```

If those variables are not provided, `inference.py` automatically falls back to a deterministic local policy so the baseline still executes cleanly in bare validation environments.

The baseline policy is observation-driven rather than task-ID hardcoded. It:

- identifies likely actionable emails from inbox summaries
- infers constraints from the selected thread
- requests clarification when required fields are missing
- views only relevant participant calendars
- searches for safe booking slots
- reschedules movable conflicts
- escalates when protected conflicts block every viable option

## Reproducible local baseline

For a deterministic end-to-end local check, this repo includes an OpenAI-compatible mock server in `scripts/mock_openai_server.py`. It returns a fixed sequence of valid actions for each task so `inference.py` can be verified with the OpenAI client before you plug in a real model endpoint.

Start the mock server:

```bash
python scripts/mock_openai_server.py
```

Then run the baseline script with:

```bash
API_BASE_URL=http://127.0.0.1:8010/v1
MODEL_NAME=opsflow-mock
HF_TOKEN=dummy
python inference.py
```

Expected reproducible local scores with the current baseline:

- `easy_single_request_clear_slot`: `1.00`
- `medium_missing_info_with_conflict`: `1.00`
- `hard_priority_conflict_timezone`: `1.00`
- `hard_reschedule_internal_sync`: `1.00`
- `hard_escalate_protected_conflict`: `1.00`
- Average: `1.00`

## Docker

Build and run:

```bash
docker build -t opsflow-env .
docker run -p 7860:7860 opsflow-env
```

## Validation checklist

Before submission:

```bash
python -m openenv.cli validate
docker build .
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
python inference.py
```

Additional local smoke checks:

```bash
python scripts/run_submission_smoke_checks.py
python -m unittest tests.test_env tests.test_api
```
