"""High-level workflows that orchestrate multiple model calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from outlines.generator import Generator
from outlines.models import LlamaCpp, MLXLM, Transformers
from outlines.types import Choice, JsonSchema


@dataclass(frozen=True)
class JsonStep:
    """Describe a single JSON field generation step."""

    field: str
    kind: Optional[Literal["enum", "string"]] = None
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class _ResolvedStep:
    field: str
    kind: Literal["enum", "string"]
    max_tokens: Optional[int]
    enum_generator: Optional[Generator]


class MultiStepJsonGenerator:
    """Generate a JSON object with multiple sequential steps.

    Each step fills one field at a time. Enum steps use constrained decoding,
    while string steps are generated with a token budget. If a string step does
    not define ``max_tokens``, the schema ``maxLength`` value is used.
    The generator currently supports string prompts only.
    """

    def __init__(
        self,
        model: Any,
        output_type: Any,
        *,
        steps: Sequence[JsonStep | dict | str],
        backend: Optional[str] = None,
        prompt_separator: str = "\n",
    ) -> None:
        if not isinstance(model, (Transformers, LlamaCpp, MLXLM)):
            raise TypeError(
                "MultiStepJsonGenerator requires a steerable model (Transformers, "
                "LlamaCpp, or MLXLM)."
            )
        if not isinstance(prompt_separator, str) or not prompt_separator:
            raise ValueError("prompt_separator must be a non-empty string.")
        if not steps:
            raise ValueError("steps must be a non-empty sequence.")

        schema = JsonSchema.convert_to(output_type, ["dict"])
        if not isinstance(schema, dict):
            raise ValueError("output_type could not be converted to a schema.")
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Schema must define properties for fields.")

        self.model = model
        self.output_type = output_type
        self.backend = backend
        self.prompt_separator = prompt_separator
        self._steps = self._resolve_steps(steps, properties)

    def __call__(
        self,
        prompt: str,
        *,
        step_kwargs: Optional[dict] = None,
        **inference_kwargs: Any,
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError("MultiStepJsonGenerator only supports string prompts.")
        if step_kwargs is not None and not isinstance(step_kwargs, dict):
            raise TypeError("step_kwargs must be a dict mapping fields to kwargs.")

        step_kwargs = step_kwargs or {}
        completed: list[tuple[str, Any]] = []

        for step in self._steps:
            step_params = dict(inference_kwargs)
            step_overrides = step_kwargs.get(step.field)
            if step_overrides is not None:
                if not isinstance(step_overrides, dict):
                    raise TypeError(
                        f"step_kwargs for '{step.field}' must be a dict."
                    )
                step_params.update(step_overrides)

            if step.kind == "string":
                if (
                    "max_tokens" not in step_params
                    and "max_new_tokens" not in step_params
                ):
                    if isinstance(self.model, Transformers):
                        step_params["max_new_tokens"] = step.max_tokens
                    else:
                        step_params["max_tokens"] = step.max_tokens

            prefix = self._build_prefix(completed, step.field)
            step_prompt = self._join_prompt(prompt, prefix)

            if step.kind == "enum":
                if step.enum_generator is None:
                    raise ValueError("Enum generator is missing for step.")
                value = step.enum_generator(step_prompt, **step_params)
            else:
                value = self.model(step_prompt, **step_params)

            if isinstance(value, list):
                raise ValueError(
                    f"Step for '{step.field}' returned multiple values."
                )

            completed.append((step.field, value))

        result = {field: value for field, value in completed}
        return json.dumps(result, ensure_ascii=True)

    def _build_prefix(self, completed: list[tuple[str, Any]], field: str) -> str:
        chunks: list[str] = ["{"]
        for idx, (name, value) in enumerate(completed):
            if idx > 0:
                chunks.append(", ")
            chunks.append(
                f"{json.dumps(name, ensure_ascii=True)}:"
                f"{json.dumps(value, ensure_ascii=True)}"
            )
        if completed:
            chunks.append(", ")
        chunks.append(f"{json.dumps(field, ensure_ascii=True)}: \"")
        return "".join(chunks)

    def _join_prompt(self, prompt: str, suffix: str) -> str:
        if prompt.endswith(self.prompt_separator):
            return f"{prompt}{suffix}"
        return f"{prompt}{self.prompt_separator}{suffix}"

    def _resolve_steps(
        self,
        steps: Sequence[JsonStep | dict | str],
        properties: dict,
    ) -> list[_ResolvedStep]:
        resolved: list[_ResolvedStep] = []
        seen_fields: set[str] = set()

        for step in steps:
            if isinstance(step, JsonStep):
                field = step.field
                kind = step.kind
                max_tokens = step.max_tokens
            elif isinstance(step, str):
                field = step
                kind = None
                max_tokens = None
            elif isinstance(step, dict):
                field = step.get("field")
                kind = step.get("kind")
                max_tokens = step.get("max_tokens")
            else:
                raise TypeError("Each step must be a JsonStep, dict, or str.")

            if not isinstance(field, str) or not field:
                raise ValueError("Each step field must be a non-empty string.")
            if field in seen_fields:
                raise ValueError(f"Field '{field}' appears multiple times.")
            seen_fields.add(field)

            field_schema = properties.get(field)
            if not isinstance(field_schema, dict):
                raise ValueError(f"Field '{field}' must exist in the schema.")

            if kind is None:
                if "enum" in field_schema:
                    kind = "enum"
                elif field_schema.get("type") == "string":
                    kind = "string"
                else:
                    raise ValueError(
                        f"Field '{field}' must be an enum or string step."
                    )

            if kind == "enum":
                enum_values = field_schema.get("enum")
                if not isinstance(enum_values, list) or not enum_values:
                    raise ValueError(
                        f"Field '{field}' enum must be a non-empty list."
                    )
                if not all(isinstance(value, str) for value in enum_values):
                    raise ValueError(
                        f"Field '{field}' enum values must be strings."
                    )
                enum_generator = Generator(
                    self.model,
                    Choice(enum_values),
                    self.backend,
                )
                resolved.append(
                    _ResolvedStep(
                        field=field,
                        kind="enum",
                        max_tokens=None,
                        enum_generator=enum_generator,
                    )
                )
                continue

            if kind == "string":
                if field_schema.get("type") != "string":
                    raise ValueError(f"Field '{field}' must be a string field.")
                if max_tokens is None:
                    max_tokens = field_schema.get("maxLength")
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    raise ValueError(
                        f"Field '{field}' must define max_tokens or maxLength."
                    )
                resolved.append(
                    _ResolvedStep(
                        field=field,
                        kind="string",
                        max_tokens=max_tokens,
                        enum_generator=None,
                    )
                )
                continue

            raise ValueError("Step kind must be 'enum' or 'string'.")

        return resolved


class TwoStepJsonGenerator:
    """Generate a JSON object in two steps."""

    def __init__(
        self,
        model: Any,
        output_type: Any,
        *,
        enum_field: str,
        summary_field: str,
        summary_max_tokens: Optional[int] = None,
        backend: Optional[str] = None,
        prompt_separator: str = "\n",
    ) -> None:
        steps = [
            JsonStep(field=enum_field, kind="enum"),
            JsonStep(
                field=summary_field,
                kind="string",
                max_tokens=summary_max_tokens,
            ),
        ]
        self._generator = MultiStepJsonGenerator(
            model,
            output_type,
            steps=steps,
            backend=backend,
            prompt_separator=prompt_separator,
        )

    def __call__(
        self,
        prompt: str,
        *,
        stage1_kwargs: Optional[dict] = None,
        stage2_kwargs: Optional[dict] = None,
        **inference_kwargs: Any,
    ) -> str:
        step_kwargs = {}
        if stage1_kwargs:
            step_kwargs[self._generator._steps[0].field] = stage1_kwargs
        if stage2_kwargs:
            step_kwargs[self._generator._steps[1].field] = stage2_kwargs
        return self._generator(
            prompt,
            step_kwargs=step_kwargs,
            **inference_kwargs,
        )