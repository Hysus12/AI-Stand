from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from spbce.metrics.distributions import js_divergence, normalize_distribution
from spbce.schema.api import PredictSurveyRequest
from spbce.schema.canonical import SurveyRecord
from spbce.settings import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    initialize_runtime_env,
)
from spbce.utils.text import simple_tokenize

AutoModelForSeq2SeqLM: Any | None
AutoTokenizer: Any | None
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


PROMPT_TEMPLATES = {
    "direct": (
        "You are simulating one member of the target population.\n"
        "Population: {population}\n"
        "Question: {question}\n"
        "Options: {options}\n"
        "Reply with exactly one option text."
    ),
    "deliberative": (
        "Act as a survey respondent from the target population.\n"
        "Think briefly, then answer with one option only.\n"
        "Population: {population}\n"
        "Question: {question}\n"
        "Options: {options}"
    ),
    "strict": (
        "Answer this survey question as one person from the target population.\n"
        "Do not explain.\n"
        "Output exactly one of these options and nothing else:\n"
        "{options}\n"
        "Population: {population}\n"
        "Question: {question}"
    ),
}

MODEL_NAME_ALIASES = {
    "minimax-m2.5": "MiniMax-M2.5",
    "minimax_m2.5": "MiniMax-M2.5",
    "minimax_m25": "MiniMax-M2.5",
    "minimax-m25": "MiniMax-M2.5",
    "minimax_m2_5": "MiniMax-M2.5",
    "minimaxm25": "MiniMax-M2.5",
}

PRICING_USD_PER_1M_TOKENS: dict[tuple[str, str], tuple[float, float]] = {
    ("openai_compatible", "deepseek-chat"): (0.28, 0.42),
    ("openai", "gpt-5.1"): (1.25, 10.0),
}


@dataclass(slots=True)
class LocalLlmPersonaBaseline:
    model_name: str = "google/flan-t5-small"
    provider: str = "auto"
    env_file: str | None = None
    anthropic_base_url: str | None = None
    openai_base_url: str | None = None
    num_samples: int = 24
    few_shot_k: int = 2
    random_state: int = 42
    top_p: float = 0.95
    max_new_tokens: int = 16
    thinking_enabled: bool = False
    reasoning_effort: str = "none"
    request_timeout_seconds: float = 60.0
    _model: object | None = field(init=False, default=None, repr=False)
    _tokenizer: object | None = field(init=False, default=None, repr=False)
    _retriever: TfidfVectorizer | None = field(init=False, default=None, repr=False)
    _retriever_matrix: Any | None = field(init=False, default=None, repr=False)
    _records: list[SurveyRecord] = field(init=False, default_factory=list, repr=False)
    pool_summary: dict[str, Any] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.model_name = self._normalize_model_name(self.model_name)

    def fit(self, records: list[SurveyRecord]) -> LocalLlmPersonaBaseline:
        initialize_runtime_env(self.env_file)
        self._records = records
        corpus = [
            f"{record.question_text} Population: {record.population_text}" for record in records
        ]
        self._retriever = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
        self._retriever_matrix = self._retriever.fit_transform(corpus)
        return self

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        return MODEL_NAME_ALIASES.get(model_name.strip().lower(), model_name)

    def _provider_name(self) -> str:
        if self.provider != "auto":
            return self.provider
        if self.model_name == "MiniMax-M2.5":
            return "anthropic_compatible"
        return "local"

    def _resolved_base_url(self) -> str | None:
        provider_name = self._provider_name()
        if provider_name == "anthropic_compatible":
            return (
                self.anthropic_base_url
                or os.getenv("ANTHROPIC_BASE_URL")
                or DEFAULT_ANTHROPIC_BASE_URL
            ).rstrip("/")
        if provider_name == "openai_compatible":
            return (
                self.openai_base_url
                or os.getenv("DEEPSEEK_BASE_URL")
                or DEFAULT_DEEPSEEK_BASE_URL
            ).rstrip("/")
        if provider_name == "openai":
            return (
                self.openai_base_url
                or os.getenv("OPENAI_BASE_URL")
                or DEFAULT_OPENAI_BASE_URL
            ).rstrip("/")
        return None

    def generation_config(self) -> dict[str, Any]:
        return {
            "provider": self._provider_name(),
            "model": self.model_name,
            "base_url": self._resolved_base_url(),
            "temperature_base": 0.9,
            "temperature_step": 0.05,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "thinking_sent": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "num_samples_per_prompt": self.num_samples,
        }

    def _pricing_key(self) -> tuple[str, str]:
        return self._provider_name(), self.model_name

    def _estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = PRICING_USD_PER_1M_TOKENS.get(self._pricing_key(), (0.0, 0.0))
        return (input_tokens * input_rate / 1_000_000) + (
            output_tokens * output_rate / 1_000_000
        )

    @staticmethod
    def _extract_usage_tokens(payload: dict[str, Any]) -> tuple[int, int, str]:
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            return 0, 0, "missing"
        input_tokens = int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("prompt_cache_miss_tokens")
            or 0
        )
        output_tokens = int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            or 0
        )
        return input_tokens, output_tokens, "provider_usage"

    def _lazy_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
                raise RuntimeError("transformers seq2seq model classes are unavailable")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return self._model, self._tokenizer

    def _select_exemplars(self, request: PredictSurveyRequest) -> list[SurveyRecord]:
        if not self._records or self._retriever is None or self._retriever_matrix is None:
            return []
        query = self._retriever.transform(
            [f"{request.question_text} Population: {request.population_text}"]
        )
        similarities = cosine_similarity(query, self._retriever_matrix)[0]
        indices = np.argsort(similarities)[::-1][: self.few_shot_k]
        return [self._records[index] for index in indices if similarities[index] > 0]

    def _build_prompt(
        self,
        request: PredictSurveyRequest,
        template_name: str,
        few_shot: bool,
    ) -> str:
        options_text = "; ".join(request.options)
        prompt = PROMPT_TEMPLATES[template_name].format(
            population=request.population_text,
            question=request.question_text,
            options=options_text,
        )
        if not few_shot:
            return prompt
        exemplars = self._select_exemplars(request)
        if not exemplars:
            return prompt
        exemplar_lines = []
        for exemplar in exemplars:
            exemplar_lines.append(
                "Example:\n"
                f"Population: {exemplar.population_text}\n"
                f"Question: {exemplar.question_text}\n"
                f"Options: {'; '.join(exemplar.options)}\n"
                "Observed top option: "
                f"{exemplar.options[int(np.argmax(exemplar.observed_distribution))]}\n"
            )
        return "\n".join(exemplar_lines) + "\n" + prompt

    def _parse_persona_option(
        self,
        final_text: str,
        options: list[str],
    ) -> tuple[str | None, str, bool]:
        normalized = final_text.strip().lower()
        if not normalized:
            return None, "empty", False
        for option in options:
            if normalized == option.lower():
                return option, "exact", True
        for option in options:
            if option.lower() in normalized:
                return option, "contains_option_text", True
        raw_tokens = set(simple_tokenize(normalized))
        if not raw_tokens:
            return None, "no_tokens", False
        scored: list[tuple[int, str]] = []
        for option in options:
            option_tokens = set(simple_tokenize(option))
            overlap = len(raw_tokens.intersection(option_tokens))
            scored.append((overlap, option))
        scored.sort(reverse=True)
        best_overlap, best_option = scored[0]
        if best_overlap > 0:
            return best_option, "token_overlap", True
        return None, "unmatched", False

    def _generate_responses_local(
        self,
        prompt: str,
        num_samples: int,
        temperature: float,
    ) -> list[dict[str, Any]]:
        model, tokenizer = self._lazy_model()
        encoded = tokenizer(
            [prompt] * num_samples,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            generated = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                do_sample=True,
                top_p=self.top_p,
                temperature=temperature,
                max_new_tokens=self.max_new_tokens,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [
            {
                "final_text": text.strip(),
                "thinking_text": None,
                "raw_text_for_debug": text.strip(),
                "final_text_present": bool(text.strip()),
                "used_thinking_fallback": False,
                "payload_summary": {"provider": "local"},
                "latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
                "usage_source": "not_applicable",
            }
            for text in decoded
        ]

    def _generate_responses_anthropic_compatible(
        self,
        prompt: str,
        num_samples: int,
        temperature: float,
    ) -> list[dict[str, Any]]:
        initialize_runtime_env(self.env_file)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        base_url = self._resolved_base_url()
        if base_url is None:
            raise RuntimeError("Anthropic-compatible base URL is not configured")
        url = f"{base_url}/v1/messages"
        responses: list[dict[str, Any]] = []
        with httpx.Client(timeout=self.request_timeout_seconds) as client:
            for _ in range(num_samples):
                started = time.perf_counter()
                try:
                    response = client.post(
                        url,
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "max_tokens": self.max_new_tokens,
                            "temperature": temperature,
                            "top_p": self.top_p,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    response.raise_for_status()
                except httpx.TimeoutException as exc:
                    raise RuntimeError("Anthropic-compatible request timed out") from exc
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text[:400]
                    raise RuntimeError(
                        f"Anthropic-compatible request failed with status "
                        f"{exc.response.status_code}: {detail}"
                    ) from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(
                        f"Anthropic-compatible request failed: {exc.__class__.__name__}"
                    ) from exc
                payload = response.json()
                latency_ms = (time.perf_counter() - started) * 1000.0
                input_tokens, output_tokens, usage_source = self._extract_usage_tokens(payload)
                content_blocks = payload.get("content", [])
                final_text = "\n".join(
                    block.get("text", "")
                    for block in content_blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
                thinking_text = "\n".join(
                    block.get("thinking", "")
                    for block in content_blocks
                    if isinstance(block, dict) and block.get("type") == "thinking"
                ).strip()
                responses.append(
                    {
                        "final_text": final_text or None,
                        "thinking_text": thinking_text or None,
                        "raw_text_for_debug": final_text or thinking_text or "",
                        "final_text_present": bool(final_text),
                        "used_thinking_fallback": bool(thinking_text and not final_text),
                        "payload_summary": {
                            "block_types": [
                                block.get("type")
                                for block in content_blocks
                                if isinstance(block, dict)
                            ],
                            "stop_reason": payload.get("stop_reason"),
                        },
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "estimated_cost_usd": self._estimate_cost_usd(
                            input_tokens, output_tokens
                        ),
                        "usage_source": usage_source,
                    }
                )
        return responses

    def _generate_responses_openai_compatible(
        self,
        prompt: str,
        num_samples: int,
        temperature: float,
    ) -> list[dict[str, Any]]:
        initialize_runtime_env(self.env_file)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        base_url = self._resolved_base_url()
        if base_url is None:
            raise RuntimeError("OpenAI-compatible base URL is not configured")
        url = f"{base_url}/chat/completions"
        responses: list[dict[str, Any]] = []
        with httpx.Client(timeout=self.request_timeout_seconds) as client:
            for _ in range(num_samples):
                started = time.perf_counter()
                try:
                    response = client.post(
                        url,
                        headers={
                            "authorization": f"Bearer {api_key}",
                            "content-type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature,
                            "top_p": self.top_p,
                            "max_tokens": self.max_new_tokens,
                            "stream": False,
                        },
                    )
                    response.raise_for_status()
                except httpx.TimeoutException as exc:
                    raise RuntimeError("OpenAI-compatible request timed out") from exc
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text[:400]
                    raise RuntimeError(
                        f"OpenAI-compatible request failed with status "
                        f"{exc.response.status_code}: {detail}"
                    ) from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(
                        f"OpenAI-compatible request failed: {exc.__class__.__name__}"
                    ) from exc
                payload = response.json()
                latency_ms = (time.perf_counter() - started) * 1000.0
                input_tokens, output_tokens, usage_source = self._extract_usage_tokens(payload)
                choice = payload.get("choices", [{}])[0]
                message = choice.get("message", {})
                final_text = str(message.get("content") or "").strip()
                thinking_text = str(
                    message.get("reasoning_content")
                    or choice.get("reasoning_content")
                    or ""
                ).strip()
                responses.append(
                    {
                        "final_text": final_text or None,
                        "thinking_text": thinking_text or None,
                        "raw_text_for_debug": final_text or thinking_text or "",
                        "final_text_present": bool(final_text),
                        "used_thinking_fallback": bool(thinking_text and not final_text),
                        "payload_summary": {
                            "finish_reason": choice.get("finish_reason"),
                            "has_reasoning_content": bool(thinking_text),
                        },
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "estimated_cost_usd": self._estimate_cost_usd(
                            input_tokens, output_tokens
                        ),
                        "usage_source": usage_source,
                    }
                )
        return responses

    def _generate_responses_openai(
        self,
        prompt: str,
        num_samples: int,
        temperature: float,
    ) -> list[dict[str, Any]]:
        initialize_runtime_env(self.env_file)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base_url = self._resolved_base_url()
        if base_url is None:
            raise RuntimeError("OpenAI base URL is not configured")
        url = f"{base_url}/responses"
        responses: list[dict[str, Any]] = []
        with httpx.Client(timeout=self.request_timeout_seconds) as client:
            for _ in range(num_samples):
                started = time.perf_counter()
                try:
                    response = client.post(
                        url,
                        headers={
                            "authorization": f"Bearer {api_key}",
                            "content-type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "input": prompt,
                            "reasoning": {"effort": self.reasoning_effort},
                            "temperature": temperature,
                            "top_p": self.top_p,
                            "max_output_tokens": self.max_new_tokens,
                        },
                    )
                    response.raise_for_status()
                except httpx.TimeoutException as exc:
                    raise RuntimeError("OpenAI request timed out") from exc
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text[:400]
                    raise RuntimeError(
                        f"OpenAI request failed with status "
                        f"{exc.response.status_code}: {detail}"
                    ) from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(
                        f"OpenAI request failed: {exc.__class__.__name__}"
                    ) from exc
                payload = response.json()
                latency_ms = (time.perf_counter() - started) * 1000.0
                input_tokens, output_tokens, usage_source = self._extract_usage_tokens(payload)
                output_items = payload.get("output", [])
                message_texts: list[str] = []
                reasoning_texts: list[str] = []
                item_types: list[str] = []
                for item in output_items:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type") or "")
                    item_types.append(item_type)
                    if item_type == "message":
                        for block in item.get("content", []):
                            if isinstance(block, dict) and block.get("type") == "output_text":
                                message_texts.append(str(block.get("text") or ""))
                    if item_type == "reasoning":
                        for summary in item.get("summary", []):
                            if isinstance(summary, dict):
                                reasoning_texts.append(str(summary.get("text") or ""))
                final_text = "\n".join(text for text in message_texts if text).strip()
                thinking_text = "\n".join(text for text in reasoning_texts if text).strip()
                responses.append(
                    {
                        "final_text": final_text or None,
                        "thinking_text": thinking_text or None,
                        "raw_text_for_debug": final_text or thinking_text or "",
                        "final_text_present": bool(final_text),
                        "used_thinking_fallback": bool(thinking_text and not final_text),
                        "payload_summary": {
                            "output_types": item_types,
                            "status": payload.get("status"),
                        },
                        "latency_ms": latency_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "estimated_cost_usd": self._estimate_cost_usd(
                            input_tokens, output_tokens
                        ),
                        "usage_source": usage_source,
                    }
                )
        return responses

    def _generate_responses(
        self,
        prompt: str,
        num_samples: int,
        temperature: float,
    ) -> list[dict[str, Any]]:
        provider_name = self._provider_name()
        if provider_name == "anthropic_compatible":
            return self._generate_responses_anthropic_compatible(prompt, num_samples, temperature)
        if provider_name == "openai_compatible":
            return self._generate_responses_openai_compatible(prompt, num_samples, temperature)
        if provider_name == "openai":
            return self._generate_responses_openai(prompt, num_samples, temperature)
        return self._generate_responses_local(prompt, num_samples, temperature)

    def sample_distribution(
        self,
        request: PredictSurveyRequest,
        few_shot: bool,
        template_names: list[str] | None = None,
        num_samples: int | None = None,
    ) -> dict[str, Any]:
        templates = template_names or list(PROMPT_TEMPLATES)
        draws_per_template = num_samples or self.num_samples
        aggregate_counts = {option: 0 for option in request.options}
        template_distributions: dict[str, list[float] | None] = {}
        raw_response_examples: list[dict[str, Any]] = []
        raw_responses_all: list[str] = []
        per_template_diversity: dict[str, dict[str, Any]] = {}
        total_samples = 0
        final_text_present_count = 0
        used_thinking_fallback_count = 0
        invalid_output_count = 0
        valid_sample_count = 0
        parser_failure_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_estimated_cost_usd = 0.0
        request_latencies_ms: list[float] = []

        for offset, template_name in enumerate(templates):
            prompt = self._build_prompt(request, template_name=template_name, few_shot=few_shot)
            temperature = 0.9 + (0.05 * offset)
            outputs = self._generate_responses(
                prompt,
                num_samples=draws_per_template,
                temperature=temperature,
            )
            counts = {option: 0 for option in request.options}
            valid_template_count = 0
            template_raw = [
                str(output.get("raw_text_for_debug") or "").strip() for output in outputs
            ]
            raw_responses_all.extend(template_raw)
            unique_template_responses = len({text for text in template_raw if text})
            per_template_diversity[template_name] = {
                "temperature": temperature,
                "total_samples": len(template_raw),
                "unique_raw_responses": unique_template_responses,
                "all_samples_unique": bool(template_raw)
                and unique_template_responses == len(template_raw),
            }
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
                parsed_option: str | None = None
                parser_method = "missing_final_text"
                is_valid = False
                if final_text_present:
                    parsed_option, parser_method, is_valid = self._parse_persona_option(
                        final_text,
                        request.options,
                    )
                if is_valid and parsed_option is not None:
                    counts[parsed_option] += 1
                    aggregate_counts[parsed_option] += 1
                    valid_sample_count += 1
                    valid_template_count += 1
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
                            "raw_response": str(output.get("raw_text_for_debug") or ""),
                            "final_text": final_text or None,
                            "thinking_text": thinking_text or None,
                            "final_text_present": final_text_present,
                            "used_thinking_fallback": used_thinking_fallback,
                            "payload_summary": output.get("payload_summary"),
                            "parser_method": parser_method,
                            "invalid_output": not is_valid,
                            "parsed_option": parsed_option,
                        }
                    )
            template_distributions[template_name] = (
                normalize_distribution([counts[option] for option in request.options]).tolist()
                if valid_template_count
                else None
            )

        valid_distributions = [
            distribution
            for distribution in template_distributions.values()
            if distribution is not None
        ]
        mean_distribution = (
            normalize_distribution(
                [aggregate_counts[option] for option in request.options]
            ).tolist()
            if valid_sample_count
            else None
        )
        pairwise_js = []
        for first_index in range(len(valid_distributions)):
            for second_index in range(first_index + 1, len(valid_distributions)):
                pairwise_js.append(
                    js_divergence(
                        valid_distributions[first_index],
                        valid_distributions[second_index],
                    )
                )
        variance = (
            float(np.mean(np.var(np.asarray(valid_distributions, dtype=float), axis=0)))
            if valid_distributions
            else 0.0
        )
        unique_raw_responses = len({text for text in raw_responses_all if text})
        return {
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
            "json_compliance_rate": 0.0,
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

    def predict_proba(self, request: PredictSurveyRequest, few_shot: bool = False) -> list[float]:
        result = self.sample_distribution(request, few_shot=few_shot)
        distribution = result["distribution"]
        if distribution is None:
            return normalize_distribution(np.ones(len(request.options), dtype=float)).tolist()
        return distribution
