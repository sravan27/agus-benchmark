from __future__ import annotations

import json
import re
from typing import Any, TypeVar


try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - Kaggle package depends on pydantic at runtime.
    BaseModel = object  # type: ignore[assignment]


T = TypeVar("T", bound=BaseModel)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _strip_control_characters(raw_text: str) -> str:
    return _CONTROL_CHAR_RE.sub("", raw_text).strip()


def _dedupe_candidates(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _extract_json_object_candidates(raw_text: str) -> list[str]:
    text = _strip_control_characters(raw_text)
    candidates: list[str] = []

    for match in _CODE_FENCE_RE.finditer(text):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)

    start_index: int | None = None
    depth = 0
    in_string = False
    escaping = False

    for index, char in enumerate(text):
        if start_index is None:
            if char == "{":
                start_index = index
                depth = 1
                in_string = False
                escaping = False
            continue

        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start_index : index + 1].strip())
                start_index = None

    return _dedupe_candidates(candidates)


def _validate_schema(payload: dict[str, Any], schema: type[T]) -> T:
    if hasattr(schema, "model_validate"):
        return schema.model_validate(payload)  # type: ignore[return-value]
    return schema.parse_obj(payload)  # type: ignore[attr-defined, return-value]


def parse_structured_response(raw_text: str, schema: type[T]) -> T:
    candidates = _extract_json_object_candidates(raw_text)
    validation_errors: list[str] = []

    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            validation_errors.append(f"json decode failed: {exc}")
            continue

        if not isinstance(payload, dict):
            validation_errors.append("decoded payload was not a JSON object")
            continue

        try:
            return _validate_schema(payload, schema)
        except Exception as exc:  # pragma: no cover - schema failure text is environment-specific.
            validation_errors.append(f"schema validation failed: {exc}")

    excerpt = _strip_control_characters(raw_text)[:400]
    raise ValueError(
        "Could not recover a valid JSON object matching "
        f"{schema.__name__}. Raw response excerpt: {excerpt!r}. "
        f"Validation attempts: {validation_errors[-3:]}"
    )


def prompt_for_schema(llm, prompt: str, schema: type[T]) -> T:
    raw_response = llm.prompt(prompt, schema=str)
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)
    return parse_structured_response(raw_response, schema)
