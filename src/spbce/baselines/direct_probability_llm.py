from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spbce.baselines.persona_llm import LocalLlmPersonaBaseline
from spbce.metrics.distributions import js_divergence, normalize_distribution
from spbce.schema.api import PredictSurveyRequest
from spbce.utils.text import simple_tokenize

DIRECT_PROBABILITY_PROMPT_TEMPLATES = {
    "probability_json_v1": (
        "Estimate the probability that the target population selects each option.\n"
        "Population: {population}\n"
        "Question: {question}\n"
        "Options: {options}\n"
        "Return JSON only. Use option text as keys and probabilities between 0 and 1 as values."
    ),
    "probability_json_v2": (
        "Predict the option probability distribution for this survey item.\n"
        "Target population: {population}\n"
        "Question: {question}\n"
        "Options: {options}\n"
        "Respond with a JSON object whose keys are the exact option texts "
        "and whose values sum to 1."
    ),
    "probability_json_v3": (
        "Provide calibrated option probabilities, not a persona answer.\n"
        "Population: {population}\n"
        "Survey question: {question}\n"
        "Answer options: {options}\n"
        "Output JSON only. Probabilities must total 1 across all listed options."
    ),
}


@dataclass(slots=True)
class DirectProbabilityLlmBaseline(LocalLlmPersonaBaseline):
    strict_json_only: bool = False
    cache_enabled: bool = True
    _result_cache: dict[str, dict[str, Any]] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        LocalLlmPersonaBaseline.__post_init__(self)

    def generation_config(self) -> dict[str, Any]:
        return super(DirectProbabilityLlmBaseline, self).generation_config() | {
            "temperature_base": 0.2,
            "temperature_step": 0.05,
            "prompt_mode": "direct_option_probabilities",
            "strict_json_only": self.strict_json_only,
            "response_schema_mode": (
                "exact_option_key_map" if self.strict_json_only else "best_effort"
            ),
        }

    def _build_probability_prompt(self, request: PredictSurveyRequest, template_name: str) -> str:
        options_text = "; ".join(request.options)
        prompt = DIRECT_PROBABILITY_PROMPT_TEMPLATES[template_name].format(
            population=request.population_text,
            question=request.question_text,
            options=options_text,
        )
        if self.strict_json_only:
            prompt += (
                "\nReturn exactly one JSON object."
                "\nUse every option text exactly once as a key."
                "\nUse numeric probabilities only."
                "\nDo not include markdown, prose, or code fences."
            )
        return prompt

    @staticmethod
    def _coerce_probability(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().rstrip(",")
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
                try:
                    return float(cleaned) / 100.0
                except ValueError:
                    return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_json_candidates(raw_text: str) -> list[str]:
        candidates = [raw_text.strip()]
        if "{" in raw_text and "}" in raw_text:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            candidates.insert(0, raw_text[start:end].strip())
        return candidates

    @staticmethod
    def _match_option_key(raw_key: str, options: list[str]) -> str | None:
        normalized_key = raw_key.strip().lower()
        for option in options:
            if normalized_key == option.lower():
                return option
        for option in options:
            if normalized_key in option.lower() or option.lower() in normalized_key:
                return option
        key_tokens = set(simple_tokenize(normalized_key))
        scored: list[tuple[int, str]] = []
        for option in options:
            option_tokens = set(simple_tokenize(option))
            scored.append((len(key_tokens.intersection(option_tokens)), option))
        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]
        return None

    def _parse_strict_json_distribution(
        self,
        final_text: str,
        options: list[str],
    ) -> tuple[list[float] | None, str, bool, bool]:
        for candidate in self._extract_json_candidates(final_text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                if set(payload) != set(options) or len(payload) != len(options):
                    return None, "json_schema_mismatch", False, False
                probabilities: list[float] = []
                for option in options:
                    probability = self._coerce_probability(payload.get(option))
                    if probability is None or probability < 0:
                        return None, "json_invalid_probability_value", False, False
                    probabilities.append(probability)
                if sum(probabilities) <= 0:
                    return None, "json_zero_mass", False, False
                distribution = normalize_distribution(probabilities).tolist()
                return distribution, "json_schema_exact", True, True
        return None, "invalid_json", False, False

    def _parse_distribution_response(
        self,
        final_text: str,
        options: list[str],
    ) -> tuple[list[float] | None, str, bool, bool]:
        if self.strict_json_only:
            return self._parse_strict_json_distribution(final_text, options)

        for candidate in self._extract_json_candidates(final_text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                scores = {option: 0.0 for option in options}
                for raw_key, raw_value in payload.items():
                    matched_option = self._match_option_key(str(raw_key), options)
                    probability = self._coerce_probability(raw_value)
                    if matched_option is not None and probability is not None:
                        scores[matched_option] = max(probability, 0.0)
                if any(value > 0 for value in scores.values()):
                    distribution = normalize_distribution(
                        [scores[option] for option in options]
                    ).tolist()
                    return distribution, "json", True, True
                return None, "json_invalid_option_payload", False, True

        regex_scores = {option: 0.0 for option in options}
        for option in options:
            pattern = re.compile(
                rf"{re.escape(option)}\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?%?)",
                flags=re.IGNORECASE,
            )
            match = pattern.search(final_text)
            if match:
                probability = self._coerce_probability(match.group(1))
                if probability is not None:
                    regex_scores[option] = max(probability, 0.0)
        if any(value > 0 for value in regex_scores.values()):
            distribution = normalize_distribution(
                [regex_scores[option] for option in options]
            ).tolist()
            return distribution, "regex", True, False

        return None, "invalid_format", False, False

    def sample_distribution(
        self,
        request: PredictSurveyRequest,
        few_shot: bool = False,
        template_names: list[str] | None = None,
        num_samples: int | None = None,
    ) -> dict[str, Any]:
        del few_shot
        cache_key = "||".join(
            [
                request.question_text.strip().lower(),
                " / ".join(option.strip().lower() for option in request.options),
                request.population_text.strip().lower(),
            ]
        )
        if self.cache_enabled and cache_key in self._result_cache:
            return copy.deepcopy(self._result_cache[cache_key])
        templates = template_names or list(DIRECT_PROBABILITY_PROMPT_TEMPLATES)
        draws_per_template = num_samples or self.num_samples
        aggregate_scores = np.zeros(len(request.options), dtype=float)
        template_distributions: dict[str, list[float] | None] = {}
        raw_response_examples: list[dict[str, Any]] = []
        raw_responses_all: list[str] = []
        per_template_diversity: dict[str, dict[str, Any]] = {}
        sample_distributions: list[list[float]] = []
        total_samples = 0
        final_text_present_count = 0
        used_thinking_fallback_count = 0
        invalid_output_count = 0
        json_compliance_count = 0
        parser_failure_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_estimated_cost_usd = 0.0
        request_latencies_ms: list[float] = []

        for offset, template_name in enumerate(templates):
            prompt = self._build_probability_prompt(request, template_name=template_name)
            temperature = 0.2 + (0.05 * offset)
            template_samples: list[list[float]] = []
            template_raw: list[str] = []
            outputs = self._generate_responses(
                prompt,
                num_samples=draws_per_template,
                temperature=temperature,
            )
            for sample_index, output in enumerate(outputs):
                total_samples += 1
                final_text = str(output.get("final_text") or "").strip()
                thinking_text = str(output.get("thinking_text") or "").strip()
                final_text_present = bool(output.get("final_text_present"))
                used_thinking_fallback = bool(output.get("used_thinking_fallback"))
                total_input_tokens += int(output.get("input_tokens") or 0)
                total_output_tokens += int(output.get("output_tokens") or 0)
                total_estimated_cost_usd += float(output.get("estimated_cost_usd") or 0.0)
                request_latencies_ms.append(float(output.get("latency_ms") or 0.0))
                if final_text_present:
                    final_text_present_count += 1
                if used_thinking_fallback:
                    used_thinking_fallback_count += 1
                debug_text = str(output.get("raw_text_for_debug") or "").strip()
                template_raw.append(debug_text)
                parsed_distribution: list[float] | None = None
                parser_method = "missing_final_text"
                is_valid = False
                json_compliant = False
                if final_text_present:
                    parsed_distribution, parser_method, is_valid, json_compliant = (
                        self._parse_distribution_response(
                            final_text,
                            request.options,
                        )
                    )
                if json_compliant:
                    json_compliance_count += 1
                if is_valid and parsed_distribution is not None:
                    template_samples.append(parsed_distribution)
                    sample_distributions.append(parsed_distribution)
                    aggregate_scores += np.asarray(parsed_distribution, dtype=float)
                else:
                    invalid_output_count += 1
                    if final_text_present:
                        parser_failure_count += 1
                if len(raw_response_examples) < 20:
                    raw_response_examples.append(
                        {
                            "template_name": template_name,
                            "sample_index": sample_index,
                            "temperature": temperature,
                            "raw_response": debug_text,
                            "final_text": final_text or None,
                            "thinking_text": thinking_text or None,
                            "final_text_present": final_text_present,
                            "used_thinking_fallback": used_thinking_fallback,
                            "payload_summary": output.get("payload_summary"),
                            "parser_method": parser_method,
                            "json_compliant": json_compliant,
                            "invalid_output": not is_valid,
                            "parsed_distribution": parsed_distribution,
                        }
                    )
            raw_responses_all.extend(template_raw)
            unique_template_responses = len({text for text in template_raw if text})
            per_template_diversity[template_name] = {
                "temperature": temperature,
                "total_samples": len(template_raw),
                "unique_raw_responses": unique_template_responses,
                "all_samples_unique": bool(template_raw)
                and unique_template_responses == len(template_raw),
            }
            template_distributions[template_name] = (
                np.mean(np.asarray(template_samples, dtype=float), axis=0).tolist()
                if template_samples
                else None
            )

        valid_distributions = [
            distribution
            for distribution in template_distributions.values()
            if distribution is not None
        ]
        mean_distribution = (
            normalize_distribution(aggregate_scores).tolist() if sample_distributions else None
        )
        pairwise_js: list[float] = []
        for first_index in range(len(valid_distributions)):
            for second_index in range(first_index + 1, len(valid_distributions)):
                pairwise_js.append(
                    js_divergence(
                        valid_distributions[first_index],
                        valid_distributions[second_index],
                    )
                )
        variance = (
            float(np.mean(np.var(np.asarray(sample_distributions, dtype=float), axis=0)))
            if sample_distributions
            else 0.0
        )
        unique_raw_responses = len({text for text in raw_responses_all if text})
        result = {
            "distribution": mean_distribution,
            "template_distributions": template_distributions,
            "sampling_variance": variance,
            "prompt_sensitivity_js": float(np.mean(pairwise_js)) if pairwise_js else 0.0,
            "prompt_paraphrase_sensitivity_js": float(np.mean(pairwise_js)) if pairwise_js else 0.0,
            "raw_response_examples": raw_response_examples,
            "raw_response_diversity": {
                "total_samples": len(raw_responses_all),
                "unique_raw_responses": unique_raw_responses,
                "all_samples_unique": bool(raw_responses_all)
                and unique_raw_responses == len(raw_responses_all),
                "per_template": per_template_diversity,
            },
            "final_text_present_rate": (
                float(final_text_present_count / total_samples) if total_samples else 0.0
            ),
            "used_thinking_fallback_rate": (
                float(used_thinking_fallback_count / total_samples) if total_samples else 0.0
            ),
            "invalid_output_rate": (
                float(invalid_output_count / total_samples) if total_samples else 0.0
            ),
            "json_compliance_rate": (
                float(json_compliance_count / total_samples) if total_samples else 0.0
            ),
            "parser_failure_rate": (
                float(parser_failure_count / total_samples) if total_samples else 0.0
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "estimated_api_cost_usd": total_estimated_cost_usd,
            "request_latencies_ms": request_latencies_ms,
            "generation_config": self.generation_config(),
            "scorable": mean_distribution is not None,
        }
        if self.cache_enabled:
            self._result_cache[cache_key] = copy.deepcopy(result)
        return result

    def predict_proba(self, request: PredictSurveyRequest, few_shot: bool = False) -> list[float]:
        del few_shot
        result = self.sample_distribution(request)
        distribution = result["distribution"]
        if distribution is None:
            return normalize_distribution(np.ones(len(request.options), dtype=float)).tolist()
        return distribution
