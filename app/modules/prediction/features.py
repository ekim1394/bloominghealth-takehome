"""Feature engineering for call outcome prediction."""

from typing import Any

from app.modules.prediction.schemas import CallEvent, CallMetadata, EventType


class FeatureEngineer:
    """Computes features from call events and metadata for ML prediction."""

    def compute_features(
        self,
        events: list[CallEvent],
        metadata: CallMetadata,
    ) -> dict[str, Any]:
        """
        Compute features from call events and metadata.

        Args:
            events: List of call events (will be sorted by timestamp).
            metadata: Call context metadata.

        Returns:
            Dictionary of computed features ready for model input.
        """
        if not events:
            return self._empty_features(metadata)

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Compute temporal features
        total_duration = self._compute_total_duration(sorted_events)
        silence_duration = self._sum_duration_by_type(sorted_events, EventType.SILENCE)
        agent_duration = self._sum_duration_by_type(sorted_events, EventType.AGENT_SPEECH)
        user_duration = self._sum_duration_by_type(sorted_events, EventType.USER_SPEECH)

        # Compute interaction features
        turn_count = self._count_turns(sorted_events)
        interruption_count = self._count_interruptions(sorted_events)
        tool_usage_count = sum(1 for e in sorted_events if e.type == EventType.TOOL_CALL)

        # Build feature dict with safe division
        features: dict[str, Any] = {
            # Temporal features
            "total_duration": total_duration,
            "silence_ratio": self._safe_divide(silence_duration, total_duration),
            "agent_talk_ratio": self._safe_divide(agent_duration, total_duration),
            "user_talk_ratio": self._safe_divide(user_duration, total_duration),
            # Interaction features
            "turn_count": turn_count,
            "interruption_count": interruption_count,
            "tool_usage_count": tool_usage_count,
            # Derived ratios
            "agent_user_ratio": self._safe_divide(agent_duration, user_duration),
            "event_count": len(sorted_events),
        }

        # Flatten metadata for categorical model support
        features.update(self._flatten_metadata(metadata))

        return features

    def _compute_total_duration(self, sorted_events: list[CallEvent]) -> float:
        """Compute total call duration from first to last event timestamp."""
        if len(sorted_events) < 2:
            return 0.0
        first_ts = sorted_events[0].timestamp
        last_ts = sorted_events[-1].timestamp
        delta = (last_ts - first_ts).total_seconds() * 1000  # Convert to ms
        return max(0.0, delta)

    def _sum_duration_by_type(
        self,
        events: list[CallEvent],
        event_type: EventType,
    ) -> float:
        """Sum durations for events of a specific type."""
        total = 0.0
        for event in events:
            if event.type == event_type and event.duration_ms is not None:
                total += event.duration_ms
        return total

    def _count_turns(self, sorted_events: list[CallEvent]) -> int:
        """Count speaker transitions (agent <-> user switches)."""
        speech_types = {EventType.AGENT_SPEECH, EventType.USER_SPEECH}
        speech_events = [e for e in sorted_events if e.type in speech_types]

        if len(speech_events) < 2:
            return 0

        turn_count = 0
        prev_type = speech_events[0].type

        for event in speech_events[1:]:
            if event.type != prev_type:
                turn_count += 1
                prev_type = event.type

        return turn_count

    def _count_interruptions(self, sorted_events: list[CallEvent]) -> int:
        """
        Count interruptions: user speaking while agent is still speaking.

        An interruption occurs when a user_speech event starts before
        the previous agent_speech event ends (based on timestamp + duration).
        """
        interruption_count = 0
        active_agent_end_ms: float | None = None

        for event in sorted_events:
            event_start_ms = event.timestamp.timestamp() * 1000

            if event.type == EventType.AGENT_SPEECH:
                # Track when agent speech ends
                if event.duration_ms is not None:
                    active_agent_end_ms = event_start_ms + event.duration_ms
                else:
                    # No duration means we can't track overlap
                    active_agent_end_ms = None

            elif event.type == EventType.USER_SPEECH:
                # Check if user is speaking while agent was still speaking
                if active_agent_end_ms is not None and event_start_ms < active_agent_end_ms:
                    interruption_count += 1
                # Reset since user is now speaking
                active_agent_end_ms = None

        return interruption_count

    def _flatten_metadata(self, metadata: CallMetadata) -> dict[str, Any]:
        """Flatten metadata into feature dict for categorical encoding."""
        return {
            "agent_id": metadata.agent_id,
            "org_id": metadata.org_id,
            "time_of_day": metadata.time_of_day,
            "day_of_week": metadata.day_of_week,
            "call_purpose": metadata.call_purpose,
        }

    def _empty_features(self, metadata: CallMetadata) -> dict[str, Any]:
        """Return feature dict with zeros when no events present."""
        features = {
            "total_duration": 0.0,
            "silence_ratio": 0.0,
            "agent_talk_ratio": 0.0,
            "user_talk_ratio": 0.0,
            "turn_count": 0,
            "interruption_count": 0,
            "tool_usage_count": 0,
            "agent_user_ratio": 0.0,
            "event_count": 0,
        }
        features.update(self._flatten_metadata(metadata))
        return features

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Safe division that returns 0.0 when denominator is zero."""
        if denominator == 0:
            return 0.0
        return numerator / denominator
