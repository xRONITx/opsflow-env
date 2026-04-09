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

OpsFlowEnv turns that workflow into a deterministic benchmark with explicit step-by-step actions.

## OpenEnv compliance

The project provides:

- typed `Action`, `Observation`, and state models with Pydantic
- `reset()`, `step()`, and `state()` APIs
- `openenv.yaml`
- an HTTP server exposing `/reset`, `/step`, and `/state`
- a working `Dockerfile`
- a root-level `inference.py` using the OpenAI client and the required environment variables

## Tasks

The benchmark includes three graded tasks.

### 1. `easy_single_request_clear_slot`

- One actionable email thread
- Complete meeting information
- One obvious valid slot
- Expected resolution: book the correct meeting

### 2. `medium_missing_info_with_conflict`

- Request is missing duration
- Requested window conflicts with another event
- Expected resolution: ask for missing information rather than booking prematurely

### 3. `hard_priority_conflict_timezone`

- Urgent multi-party scheduling request
- Timezone-sensitive attendees
- Protected existing events cannot be moved automatically
- Expected resolution: inspect calendars and book the only policy-compliant slot

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
- positive reward for inspecting relevant calendars
- positive reward for a valid booking or correct clarification request
- penalties for invalid actions, policy violations, conflicting bookings, and loops

## Deterministic grading

Each task has a grader that returns a deterministic score from `0.0` to `1.0` based on:

- thread selection correctness
- extraction correctness
- calendar validity
- final resolution correctness
- policy compliance
- action efficiency

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

Expected reproducible local scores with the bundled mock server:

- `easy_single_request_clear_slot`: `1.00`
- `medium_missing_info_with_conflict`: `1.00`
- `hard_priority_conflict_timezone`: `1.00`
- Average: `1.00`

## Real baseline scores

The baseline `inference.py` script was also run against the Hugging Face router using:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=deepseek-ai/DeepSeek-V3-0324`

Observed scores:

- `easy_single_request_clear_slot`: `0.20`
- `medium_missing_info_with_conflict`: `0.20`
- `hard_priority_conflict_timezone`: `0.15`
- Average: `0.18`

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
