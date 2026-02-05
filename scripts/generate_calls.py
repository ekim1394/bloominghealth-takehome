#!/usr/bin/env python3
"""
Case Study 2: Synthetic Call Data Generator

Generates 500+ voice calls with realistic patterns for training a call outcome 
prediction model. Each pattern encodes causal relationships so the ML model 
can learn predictive features.

Patterns:
- Pattern A (Completed): Balanced turn-taking, tool calls, moderate duration.
- Pattern B (Abandoned): Long silences, user frustration, no tool completion.
- Pattern C (Transferred): Long duration, high complexity, transfer tool call.
- Pattern D (Error): Very short calls or immediate failures after tool use.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from faker import Faker

# ============================================================================
# Configuration Constants
# ============================================================================

AGENTS = ["agent_001", "agent_002", "agent_003", "agent_004", "agent_005"]
ORGS = ["org_alpha", "org_beta", "org_gamma"]
CALL_PURPOSES = ["sdoh_screening", "appointment_scheduling", "billing", "care_follow_up"]
PHONE_TYPES = ["mobile", "landline"]
TIMES_OF_DAY = ["morning", "afternoon", "evening"]
DAYS_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Tool names by category
COMPLETION_TOOLS = ["submit_survey", "schedule_appointment", "confirm_details", "process_payment"]
TRANSFER_TOOLS = ["transfer_agent", "escalate_supervisor"]
SCREENING_TOOLS = ["submit_survey_response", "verify_identity"]

# Outcome distribution (60% completed, 20% abandoned, 15% transferred, 5% error)
OUTCOME_WEIGHTS = {
    "completed": 0.60,
    "abandoned": 0.20,
    "transferred": 0.15,
    "error": 0.05,
}

fake = Faker()
Faker.seed(42)
random.seed(42)


# ============================================================================
# Helper Functions
# ============================================================================


def generate_metadata() -> dict[str, str]:
    """Generate random call metadata."""
    return {
        "agent_id": random.choice(AGENTS),
        "org_id": random.choice(ORGS),
        "call_purpose": random.choice(CALL_PURPOSES),
        "caller_phone_type": random.choice(PHONE_TYPES),
        "time_of_day": random.choice(TIMES_OF_DAY),
        "day_of_week": random.choice(DAYS_OF_WEEK),
    }


def create_event(ts: int, event_type: str, **kwargs: Any) -> dict[str, Any]:
    """Create a single event dict with timestamp and type."""
    event: dict[str, Any] = {"ts": ts, "type": event_type}
    event.update(kwargs)
    return event


def speech_event(ts: int, speaker: str, duration_ms: int, words: int) -> dict[str, Any]:
    """Create a speech event for agent or user."""
    return create_event(ts, f"{speaker}_speech", duration_ms=duration_ms, words=words)


def silence_event(ts: int, duration_ms: int) -> dict[str, Any]:
    """Create a silence event."""
    return create_event(ts, "silence", duration_ms=duration_ms)


def tool_event(ts: int, tool: str) -> dict[str, Any]:
    """Create a tool_call event."""
    return create_event(ts, "tool_call", tool=tool)


# ============================================================================
# Pattern Generators
# ============================================================================


def generate_completed_call() -> dict[str, Any]:
    """
    Pattern A: Happy Path -> Completed
    
    Characteristics:
    - Balanced turn-taking (agent/user/agent/user)
    - Low silence ratios (< 3s per pause)
    - Successful tool call present
    - Moderate duration (2-5 minutes, 120-300 seconds)
    """
    call_id = str(uuid.uuid4())
    metadata = generate_metadata()
    events: list[dict[str, Any]] = []
    
    # Start call
    ts = 0
    events.append(create_event(ts, "call_start"))
    ts += random.randint(1, 3)
    
    # Balanced conversational turns (3-6 exchanges)
    num_exchanges = random.randint(3, 6)
    
    for _ in range(num_exchanges):
        # Agent speaks (moderate length)
        agent_duration = random.randint(2000, 4500)
        agent_words = random.randint(25, 55)
        events.append(speech_event(ts, "agent", agent_duration, agent_words))
        ts += agent_duration // 1000 + random.randint(1, 2)
        
        # Short pause/user thinking (natural, not frustrating)
        if random.random() < 0.4:
            pause_duration = random.randint(500, 2000)
            events.append(silence_event(ts, pause_duration))
            ts += pause_duration // 1000 + 1
        
        # User responds (engaged, reasonable length)
        user_duration = random.randint(1000, 3500)
        user_words = random.randint(10, 40)
        events.append(speech_event(ts, "user", user_duration, user_words))
        ts += user_duration // 1000 + random.randint(1, 2)
    
    # Successful tool call (key indicator of completion)
    tool_name = random.choice(COMPLETION_TOOLS)
    if metadata["call_purpose"] == "sdoh_screening":
        tool_name = "submit_survey_response"
    elif metadata["call_purpose"] == "appointment_scheduling":
        tool_name = "schedule_appointment"
    
    events.append(tool_event(ts, tool_name))
    ts += random.randint(2, 5)
    
    # Closing agent speech
    events.append(speech_event(ts, "agent", random.randint(1500, 3000), random.randint(18, 35)))
    ts += random.randint(3, 6)
    
    # Final user acknowledgment
    events.append(speech_event(ts, "user", random.randint(500, 1500), random.randint(3, 12)))
    ts += random.randint(2, 4)
    
    # End call - ensure duration is 2-5 minutes
    call_duration = max(120, min(300, ts + random.randint(5, 30)))
    events.append(create_event(call_duration, "call_end"))
    
    return {
        "call_id": call_id,
        "metadata": metadata,
        "events": events,
        "outcome": "completed",
        "survey_completion_rate": round(random.uniform(0.8, 1.0), 2),
    }


def generate_abandoned_call() -> dict[str, Any]:
    """
    Pattern B: Frustrated User -> Abandoned
    
    Characteristics:
    - Long silence events (latency, user waiting)
    - Frequent short user interruptions
    - No successful tool calls
    - Ends abruptly after user speech (hang up)
    """
    call_id = str(uuid.uuid4())
    metadata = generate_metadata()
    events: list[dict[str, Any]] = []
    
    # Start call
    ts = 0
    events.append(create_event(ts, "call_start"))
    ts += random.randint(1, 3)
    
    # Initial agent greeting
    events.append(speech_event(ts, "agent", random.randint(3000, 5000), random.randint(35, 60)))
    ts += random.randint(6, 10)
    
    # Long problematic silence (system latency / confused user)
    events.append(silence_event(ts, random.randint(5000, 12000)))
    ts += random.randint(8, 15)
    
    # A few frustrated exchanges
    num_exchanges = random.randint(2, 4)
    
    for i in range(num_exchanges):
        # Agent attempts to re-engage
        events.append(speech_event(ts, "agent", random.randint(2500, 4000), random.randint(30, 50)))
        ts += random.randint(4, 7)
        
        # Another long silence (frustrating wait)
        if random.random() < 0.7:
            events.append(silence_event(ts, random.randint(4000, 10000)))
            ts += random.randint(6, 12)
        
        # Short, terse user response (frustration indicator)
        user_duration = random.randint(300, 1200)
        user_words = random.randint(1, 8)  # Very few words = frustration
        events.append(speech_event(ts, "user", user_duration, user_words))
        ts += user_duration // 1000 + random.randint(1, 3)
    
    # Final user utterance before abandoning
    events.append(speech_event(ts, "user", random.randint(200, 800), random.randint(1, 5)))
    ts += random.randint(1, 3)
    
    # Abrupt call end (no tool completion, no agent closing)
    events.append(create_event(ts, "call_end"))
    
    return {
        "call_id": call_id,
        "metadata": metadata,
        "events": events,
        "outcome": "abandoned",
        "survey_completion_rate": 0.0,
    }


def generate_transferred_call() -> dict[str, Any]:
    """
    Pattern C: Complex Case -> Transferred
    
    Characteristics:
    - Long total duration (> 8 minutes)
    - High word counts per turn (complex discussion)
    - Often billing/complex purposes
    - Ends with transfer_agent tool call
    """
    call_id = str(uuid.uuid4())
    metadata = generate_metadata()
    
    # Bias toward complex call purposes
    if random.random() < 0.6:
        metadata["call_purpose"] = random.choice(["billing", "care_follow_up"])
    
    events: list[dict[str, Any]] = []
    
    # Start call
    ts = 0
    events.append(create_event(ts, "call_start"))
    ts += random.randint(1, 3)
    
    # Long, complex discussion (6-10 exchanges with high word counts)
    num_exchanges = random.randint(6, 10)
    
    for _ in range(num_exchanges):
        # Agent speaks (longer, more explanatory)
        agent_duration = random.randint(4000, 8000)
        agent_words = random.randint(50, 100)  # High word count
        events.append(speech_event(ts, "agent", agent_duration, agent_words))
        ts += agent_duration // 1000 + random.randint(2, 4)
        
        # Normal pauses
        if random.random() < 0.5:
            events.append(silence_event(ts, random.randint(1000, 3000)))
            ts += random.randint(2, 4)
        
        # User explains complex issue (high word counts)
        user_duration = random.randint(3000, 7000)
        user_words = random.randint(35, 85)  # User also has lots to say
        events.append(speech_event(ts, "user", user_duration, user_words))
        ts += user_duration // 1000 + random.randint(2, 4)
    
    # Agent realizes complexity, needs to transfer
    events.append(speech_event(ts, "agent", random.randint(3000, 5000), random.randint(40, 60)))
    ts += random.randint(5, 8)
    
    # Transfer tool call (key indicator)
    events.append(tool_event(ts, random.choice(TRANSFER_TOOLS)))
    ts += random.randint(2, 5)
    
    # Brief hold/transfer notification
    events.append(speech_event(ts, "agent", random.randint(1500, 2500), random.randint(15, 25)))
    ts += random.randint(3, 6)
    
    # End call - ensure duration > 8 minutes (480 seconds)
    call_duration = max(480, ts + random.randint(10, 60))
    events.append(create_event(call_duration, "call_end"))
    
    return {
        "call_id": call_id,
        "metadata": metadata,
        "events": events,
        "outcome": "transferred",
        "survey_completion_rate": round(random.uniform(0.0, 0.3), 2),
    }


def generate_error_call() -> dict[str, Any]:
    """
    Pattern D: System Failure -> Error
    
    Characteristics:
    - Very short calls (< 30 seconds) OR
    - Calls ending immediately after a specific tool failure
    - Minimal meaningful interaction
    """
    call_id = str(uuid.uuid4())
    metadata = generate_metadata()
    events: list[dict[str, Any]] = []
    
    # Start call
    ts = 0
    events.append(create_event(ts, "call_start"))
    
    # Two variants of error patterns
    if random.random() < 0.5:
        # Variant 1: Immediate failure (< 10 seconds)
        ts += random.randint(1, 3)
        
        # Brief agent start
        if random.random() < 0.7:
            events.append(speech_event(ts, "agent", random.randint(500, 1500), random.randint(5, 15)))
            ts += random.randint(2, 4)
        
        # Abrupt end
        events.append(create_event(ts, "call_end"))
    else:
        # Variant 2: Tool failure causes error
        ts += random.randint(1, 3)
        
        # Initial exchange
        events.append(speech_event(ts, "agent", random.randint(2000, 3500), random.randint(25, 40)))
        ts += random.randint(4, 6)
        
        events.append(speech_event(ts, "user", random.randint(1000, 2000), random.randint(10, 20)))
        ts += random.randint(2, 4)
        
        # Some tool attempt that "fails"
        events.append(tool_event(ts, random.choice(["verify_identity", "process_payment", "check_availability"])))
        ts += random.randint(1, 3)
        
        # Immediate call termination after tool (error indicator)
        events.append(create_event(ts, "call_end"))
    
    # Ensure call is short (< 30 seconds generally)
    # (timestamps are already short from the logic above)
    
    return {
        "call_id": call_id,
        "metadata": metadata,
        "events": events,
        "outcome": "error",
        "survey_completion_rate": 0.0,
    }


# ============================================================================
# Main Generator
# ============================================================================


def generate_call(outcome_type: str) -> dict[str, Any]:
    """
    Generate a single call based on the specified outcome type.
    
    Args:
        outcome_type: One of 'completed', 'abandoned', 'transferred', 'error'
    
    Returns:
        A call dictionary with metadata, events, and outcome.
    """
    generators = {
        "completed": generate_completed_call,
        "abandoned": generate_abandoned_call,
        "transferred": generate_transferred_call,
        "error": generate_error_call,
    }
    
    if outcome_type not in generators:
        raise ValueError(f"Unknown outcome type: {outcome_type}")
    
    return generators[outcome_type]()


def generate_dataset(n_calls: int = 500) -> list[dict[str, Any]]:
    """
    Generate a dataset of n_calls with weighted outcome distribution.
    
    Distribution:
    - 60% completed
    - 20% abandoned  
    - 15% transferred
    - 5% error
    """
    calls: list[dict[str, Any]] = []
    outcomes = list(OUTCOME_WEIGHTS.keys())
    weights = list(OUTCOME_WEIGHTS.values())
    
    for _ in range(n_calls):
        outcome = random.choices(outcomes, weights=weights, k=1)[0]
        calls.append(generate_call(outcome))
    
    return calls


def main() -> None:
    """Generate dataset and save to data/calls.json."""
    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "calls.json"
    
    print("Generating 500 synthetic calls...")
    calls = generate_dataset(n_calls=500)
    
    # Count outcomes for verification
    outcome_counts = {}
    for call in calls:
        outcome = call["outcome"]
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    
    print(f"Generated {len(calls)} calls:")
    for outcome, count in sorted(outcome_counts.items()):
        pct = count / len(calls) * 100
        print(f"  - {outcome}: {count} ({pct:.1f}%)")
    
    # Save to JSON
    output_data = {"calls": calls}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
