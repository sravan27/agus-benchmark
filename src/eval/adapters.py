"""Local-first model adapters for AGUS evaluation."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from src.eval.interactive_runner import EpisodeSpec, EpisodeTurnSpec

SUPPORTED_STATIC_FAMILIES = {
    "hidden_rule",
    "shift_transfer",
    "metacog_revision",
    "attention_distractors",
    "social_miniworlds",
}


def _stable_fraction(*parts: Any) -> float:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _coerce_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _wrong_like(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) > 1:
            return list(reversed(value))
        if value:
            return [_wrong_like(value[0])]
        return value
    if isinstance(value, dict):
        mutated = dict(value)
        for key, inner in mutated.items():
            mutated[key] = _wrong_like(inner)
            break
        return mutated
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}_alt"
    return value


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload:
        return {}
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(payload[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _json_block(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def static_response_template(family: str) -> dict[str, Any]:
    """Return the expected response shape for a task family."""
    if family == "hidden_rule":
        return {
            "induction_predictions": [[0, 1, 2]],
            "shift_predictions": [[1, 2, 3]],
        }
    if family == "shift_transfer":
        return {
            "source_prediction": ["alpha", "beta"],
            "transfer_prediction": ["token_1", "token_2"],
        }
    if family == "metacog_revision":
        return {
            "initial_answer": ["candidate"],
            "initial_confidence": 0.5,
            "initial_rule_guess": "candidate_rule",
            "revised_answer": ["updated"],
            "revised_confidence": 0.8,
            "revised_rule_guess": "updated_rule",
        }
    if family == "attention_distractors":
        return {
            "prediction": [0, 1, 2],
            "selected_signal": "signal_sequence",
        }
    if family == "social_miniworlds":
        return {
            "actual_location_prediction": "box",
            "belief_prediction": "basket",
            "trust_prediction": "Alex",
        }
    raise ValueError(f"Unsupported family: {family}")


def build_static_prompt(task: dict[str, Any]) -> str:
    """Build a simple JSON-only prompt for static local evaluation."""
    schema = static_response_template(task["family"])
    return (
        "You are evaluating on AGUS, a dynamic cognition benchmark.\n"
        "Solve the task below and return ONLY one JSON object matching the response schema.\n"
        "Do not include markdown, commentary, or code fences.\n\n"
        f"Task family: {task['family']}\n"
        f"Task id: {task['task_id']}\n"
        f"Response schema:\n{_json_block(schema)}\n\n"
        f"Task payload:\n{_json_block(task)}\n"
    )


def build_interactive_prompt(spec: EpisodeSpec, turn: EpisodeTurnSpec, prior_turns: list[dict[str, Any]]) -> str:
    """Build a JSON-only prompt for one interactive turn."""
    schema = {
        "answer": turn.accepted_answers[0] if turn.accepted_answers else None,
        "confidence": 0.5,
        "rule_explanation": "brief_hypothesis",
        "evidence_acknowledged": turn.event is not None,
        "contradiction_detected": False,
        "attended_signals": ["signal_a"],
        "ignored_signals": ["distractor_b"],
        "trust_scores_by_agent": {"Agent": 0.7},
        "inferred_agent_beliefs": {"Agent": "location"},
        "revision_events": ["belief_updated"],
        "metadata": {"rule_tag": "belief_tracking"},
    }
    return (
        "You are participating in an AGUS interactive evaluation episode.\n"
        "Respond to the current turn and return ONLY one JSON object matching the response schema.\n"
        "Do not include markdown, commentary, or code fences.\n\n"
        f"Episode family: {spec.family}\n"
        f"Episode type: {spec.episode_type}\n"
        f"Task id: {spec.task_id}\n"
        f"Initial context:\n{_json_block(spec.initial_context)}\n\n"
        f"Prior turn responses:\n{_json_block(prior_turns)}\n\n"
        f"Current turn:\n{_json_block(turn.to_dict())}\n\n"
        f"Response schema:\n{_json_block(schema)}\n"
    )


def normalize_static_prediction(family: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw adapter payload into the expected static schema."""
    if family == "hidden_rule":
        return {
            "induction_predictions": list(payload.get("induction_predictions", [])),
            "shift_predictions": list(payload.get("shift_predictions", [])),
        }
    if family == "shift_transfer":
        return {
            "source_prediction": payload.get("source_prediction"),
            "transfer_prediction": payload.get("transfer_prediction"),
        }
    if family == "metacog_revision":
        return {
            "initial_answer": payload.get("initial_answer"),
            "initial_confidence": _coerce_float(payload.get("initial_confidence"), 0.0),
            "initial_rule_guess": str(payload.get("initial_rule_guess", "")),
            "revised_answer": payload.get("revised_answer"),
            "revised_confidence": _coerce_float(payload.get("revised_confidence"), 0.0),
            "revised_rule_guess": str(payload.get("revised_rule_guess", "")),
        }
    if family == "attention_distractors":
        return {
            "prediction": payload.get("prediction"),
            "selected_signal": str(payload.get("selected_signal", "")),
        }
    if family == "social_miniworlds":
        return {
            "actual_location_prediction": payload.get("actual_location_prediction"),
            "belief_prediction": payload.get("belief_prediction"),
            "trust_prediction": payload.get("trust_prediction"),
        }
    raise ValueError(f"Unsupported family: {family}")


def normalize_turn_payload(payload: dict[str, Any], turn: EpisodeTurnSpec) -> dict[str, Any]:
    """Normalize a raw interactive response payload."""
    return {
        "answer": payload.get("answer"),
        "confidence": _coerce_float(payload.get("confidence"), 0.0),
        "rule_explanation": str(payload.get("rule_explanation", "")),
        "evidence_acknowledged": bool(payload.get("evidence_acknowledged", turn.event is not None)),
        "contradiction_detected": bool(payload.get("contradiction_detected", False)),
        "attended_signals": list(payload.get("attended_signals", [])),
        "ignored_signals": list(payload.get("ignored_signals", [])),
        "trust_scores_by_agent": {
            str(agent): _coerce_float(score, 0.0)
            for agent, score in payload.get("trust_scores_by_agent", {}).items()
        },
        "inferred_agent_beliefs": dict(payload.get("inferred_agent_beliefs", {})),
        "revision_events": list(payload.get("revision_events", [])),
        "metadata": dict(payload.get("metadata", {})),
    }


class ModelAdapter(ABC):
    """Base interface for AGUS model adapters."""

    adapter_kind = "base"
    supports_interactive = True

    def __init__(self, name: str) -> None:
        self.name = name

    def describe(self) -> dict[str, Any]:
        """Return inspectable adapter metadata."""
        return {
            "name": self.name,
            "adapter_kind": self.adapter_kind,
            "supports_interactive": self.supports_interactive,
        }

    @abstractmethod
    def predict_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Produce one static prediction payload."""

    def respond_turn(
        self,
        spec: EpisodeSpec,
        turn: EpisodeTurnSpec,
        prior_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Produce one interactive turn response."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support interactive evaluation.")


class MockAdapter(ModelAdapter):
    """Deterministic adapter for local harness validation and demos."""

    adapter_kind = "mock"
    supports_interactive = True

    def __init__(self, profile: str = "noisy", seed: int = 0) -> None:
        super().__init__(name=f"mock-{profile}")
        self.profile = profile
        self.seed = seed

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update({"profile": self.profile, "seed": self.seed})
        return payload

    def _chance(self, *parts: Any) -> float:
        return _stable_fraction(self.seed, self.profile, *parts)

    def _is_oracle(self) -> bool:
        return self.profile == "oracle"

    def predict_task(self, task: dict[str, Any]) -> dict[str, Any]:
        family = task["family"]
        frac = self._chance(task["task_id"], "static")

        if family == "hidden_rule":
            induction = list(task["answer"]["induction_targets"])
            shift = list(task["answer"]["shift_targets"])
            if not self._is_oracle() and frac < (0.2 if self.profile == "noisy" else 0.55) and shift:
                shift[0] = _wrong_like(shift[0])
            return {
                "task_id": task["task_id"],
                "induction_predictions": induction,
                "shift_predictions": shift,
            }

        if family == "shift_transfer":
            transfer_target = task["answer"]["transfer_target"]
            if self._is_oracle():
                transfer_prediction = transfer_target
            elif self.profile == "shallow":
                transfer_prediction = task["answer"]["source_target"]
            else:
                transfer_prediction = transfer_target if frac >= 0.35 else _wrong_like(transfer_target)
            return {
                "task_id": task["task_id"],
                "source_prediction": task["answer"]["source_target"],
                "transfer_prediction": transfer_prediction,
            }

        if family == "metacog_revision":
            acceptable = list(task["answer"]["acceptable_initial_targets"])
            initial_answer = acceptable[0]
            revised_target = task["answer"]["revised_target"]
            should_fail_revision = (not self._is_oracle()) and frac < (0.15 if self.profile == "noisy" else 0.6)
            revised_answer = acceptable[-1] if should_fail_revision else revised_target
            initial_confidence = float(task["metadata"].get("expected_initial_certainty", 0.5))
            revised_confidence = 0.9 if not should_fail_revision else max(0.2, initial_confidence - 0.1)
            return {
                "task_id": task["task_id"],
                "initial_answer": initial_answer,
                "initial_confidence": initial_confidence,
                "initial_rule_guess": "candidate_rule",
                "revised_answer": revised_answer,
                "revised_confidence": revised_confidence,
                "revised_rule_guess": task["metadata"]["actual_rule"]["name"] if not should_fail_revision else "stale_rule",
            }

        if family == "attention_distractors":
            target = task["answer"]["target"]
            level = int(task.get("distractor_level", 0))
            follow_distractor = (
                not self._is_oracle()
                and level >= 2
                and frac < (0.3 if self.profile == "noisy" else 0.8)
            )
            prediction = task["query"]["record"]["distractor_sequence"] if follow_distractor else target
            return {
                "task_id": task["task_id"],
                "prediction": prediction,
                "selected_signal": "distractor_sequence" if follow_distractor else "signal_sequence",
            }

        if family == "social_miniworlds":
            actual = task["answer"]["actual_location"]
            belief = task["answer"]["belief_of_false_belief_agent"]
            deceiver = task["metadata"]["deceiver"]
            trust_correct = self._is_oracle() or frac >= (0.3 if self.profile == "noisy" else 0.75)
            return {
                "task_id": task["task_id"],
                "actual_location_prediction": actual,
                "belief_prediction": belief if self.profile != "shallow" else actual,
                "trust_prediction": task["answer"]["most_reliable_agent"] if trust_correct else deceiver,
            }

        raise ValueError(f"Unsupported family: {family}")

    def respond_turn(
        self,
        spec: EpisodeSpec,
        turn: EpisodeTurnSpec,
        prior_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        turn_index = len(prior_turns)
        frac = self._chance(spec.task_id, "interactive", turn_index)
        accepted = turn.accepted_answers[0] if turn.accepted_answers else None
        rule = turn.expected_rule
        if isinstance(rule, list):
            rule = rule[0]

        if self._is_oracle():
            contradiction_event = bool(
                turn.event
                and (
                    "conflict" in turn.event["event_type"]
                    or "contradiction" in turn.event["event_type"]
                )
            )
            return normalize_turn_payload(
                {
                    "answer": accepted,
                    "confidence": turn.expected_confidence,
                    "rule_explanation": str(rule or "structured_reasoning"),
                    "evidence_acknowledged": turn.event is not None,
                    "contradiction_detected": contradiction_event,
                    "attended_signals": list(turn.expected_attended_signals),
                    "ignored_signals": list(turn.expected_ignored_signals),
                    "trust_scores_by_agent": {turn.expected_trust_top: 1.0} if turn.expected_trust_top else {},
                    "inferred_agent_beliefs": dict(turn.expected_beliefs),
                    "revision_events": ["oracle_response"] if turn.event is not None else [],
                    "metadata": {"rule_tag": str(rule or "structured_reasoning")},
                },
                turn,
            )

        if spec.family in {"hidden_rule", "metacog_revision", "shift_transfer"}:
            should_update = turn_index > 0
            miss_update = self.profile == "shallow" and should_update
            answer = accepted if not miss_update else _wrong_like(accepted)
            confidence = turn.expected_confidence if not miss_update else max(0.2, turn.expected_confidence - 0.25)
            rule_tag = str(rule or "structured_reasoning")
            if miss_update:
                rule_tag = "stale_hypothesis"
            return normalize_turn_payload(
                {
                    "answer": answer,
                    "confidence": confidence,
                    "rule_explanation": rule_tag,
                    "evidence_acknowledged": turn.event is not None,
                    "contradiction_detected": bool(turn.event is not None and not miss_update),
                    "revision_events": ["belief_updated"] if should_update and not miss_update else [],
                    "metadata": {"rule_tag": rule_tag},
                },
                turn,
            )

        if spec.family == "attention_distractors":
            cue_turn = int(spec.expected["cue_turn"])
            true_target = accepted
            decoy_target = spec.expected["decoy_target"]
            pre_cue_capture = turn_index < cue_turn - 1 and frac < (0.7 if self.profile == "shallow" else 0.45)
            post_cue_recover = turn_index >= cue_turn - 1 and frac >= (0.4 if self.profile == "shallow" else 0.1)
            use_decoy = pre_cue_capture or (turn_index >= cue_turn - 1 and not post_cue_recover)
            return normalize_turn_payload(
                {
                    "answer": decoy_target if use_decoy else true_target,
                    "confidence": turn.expected_confidence if not use_decoy else max(0.2, turn.expected_confidence - 0.2),
                    "rule_explanation": "attention_tracking",
                    "evidence_acknowledged": turn.event is not None,
                    "contradiction_detected": bool(turn.event is not None and use_decoy),
                    "attended_signals": [spec.expected["trap_signal"]] if use_decoy else [spec.expected["relevant_signal"]],
                    "ignored_signals": [spec.expected["relevant_signal"]] if use_decoy else [spec.expected["trap_signal"]],
                    "revision_events": ["attention_recovery"] if turn_index >= cue_turn - 1 and not use_decoy else ["distractor_capture"] if use_decoy else [],
                    "metadata": {"rule_tag": "attention_tracking"},
                },
                turn,
            )

        reliable = spec.expected["reliable_agent"]
        deceptive = spec.expected["deceptive_agent"]
        trust_deceptive = self.profile == "shallow" and turn_index < len(spec.turns) - 1
        trust_scores = {agent: 0.35 for agent in spec.initial_context["agents"]}
        trust_scores[reliable] = 0.85 if not trust_deceptive else 0.3
        trust_scores[deceptive] = 0.15 if not trust_deceptive else 0.8
        answer = accepted
        if trust_deceptive:
            answer = {"actual_location": spec.expected["actual_location"], "trusted_agent": deceptive}
        return normalize_turn_payload(
            {
                "answer": answer,
                "confidence": turn.expected_confidence if not trust_deceptive else max(0.2, turn.expected_confidence - 0.15),
                "rule_explanation": "social_reasoning",
                "evidence_acknowledged": turn.event is not None,
                "contradiction_detected": bool(turn.event is not None),
                "trust_scores_by_agent": trust_scores,
                "inferred_agent_beliefs": dict(turn.expected_beliefs),
                "revision_events": ["trust_revised"] if not trust_deceptive and turn_index > 0 else [],
                "metadata": {"rule_tag": "social_reasoning"},
            },
            turn,
        )


class OllamaAdapter(ModelAdapter):
    """Local adapter scaffold for Ollama-style JSON inference."""

    adapter_kind = "local"
    supports_interactive = True

    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout_seconds: int = 120,
        keep_alive: str | None = "30m",
    ) -> None:
        super().__init__(name=f"ollama:{model}")
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "model": self.model,
                "host": self.host,
                "temperature": self.temperature,
                "timeout_seconds": self.timeout_seconds,
                "keep_alive": self.keep_alive,
            }
        )
        return payload

    def _build_request_payload(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _request_json(self, prompt: str) -> dict[str, Any]:
        payload = self._build_request_payload(prompt)
        request = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach the local Ollama server. Start Ollama or check --host."
            ) from exc

        text = str(raw.get("response", ""))
        parsed = _extract_json_object(text)
        if parsed:
            return parsed
        if isinstance(raw.get("response"), dict):
            return raw["response"]
        return {}

    def predict_task(self, task: dict[str, Any]) -> dict[str, Any]:
        prompt = build_static_prompt(task)
        payload = self._request_json(prompt)
        normalized = normalize_static_prediction(task["family"], payload)
        normalized["task_id"] = task["task_id"]
        return normalized

    def respond_turn(
        self,
        spec: EpisodeSpec,
        turn: EpisodeTurnSpec,
        prior_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = build_interactive_prompt(spec, turn, prior_turns)
        payload = self._request_json(prompt)
        return normalize_turn_payload(payload, turn)


def build_adapter(
    adapter_name: str,
    *,
    model: str | None = None,
    seed: int = 0,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
    timeout_seconds: int = 120,
    keep_alive: str | None = "30m",
) -> ModelAdapter:
    """Instantiate an adapter from CLI-style arguments."""
    if adapter_name.startswith("mock-"):
        profile = adapter_name.split("-", 1)[1]
        return MockAdapter(profile=profile, seed=seed)
    if adapter_name == "mock":
        return MockAdapter(profile="noisy", seed=seed)
    if adapter_name == "ollama":
        if not model:
            raise ValueError("--model is required when --adapter ollama is selected.")
        return OllamaAdapter(
            model=model,
            host=host,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
        )
    raise ValueError(f"Unsupported adapter: {adapter_name}")
