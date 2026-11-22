"""
slm_backend.py — Fast GPU Inference Backend for Cyber SLM Agent.

Implements:
- FirstJsonObjectStoppingCriteria: halts generation as soon as the first balanced
  JSON object `{...}` closes, preventing post-JSON rambling.
- Static Prompt Prefix KV-Caching: caches the tokenized & KV-encoded static system
  instruction prefix across steps so each step only prefills the dynamic state tokens.
- Greedy decoding (`do_sample=False`) with `max_new_tokens=160` (halted early on `}`).
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)


@dataclass
class SLMConfig:
    """Configuration for Small Language Model inference and agent modularity."""

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    backend: str = "huggingface"  # "huggingface" or "mock"
    quantization: Optional[str] = None  # None ("bfloat16") or "4bit"
    temperature: float = 0.0
    max_new_tokens: int = 160
    use_memory: bool = True
    enable_action_masking: bool = True
    enable_stealth_pacing: bool = True
    memory_window: int = 5
    device: str = "cuda"


class FirstJsonObjectStoppingCriteria(StoppingCriteria):
    """Stop generation immediately once the first balanced JSON object `{...}` closes."""

    def __init__(self, tokenizer: Any, prompt_len: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)

    @staticmethod
    def has_complete_json_object(text: str) -> bool:
        """Return True when `text` contains at least one balanced `{...}` object."""
        depth = 0
        in_string = False
        escaped = False
        seen_open = False
        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}" and seen_open:
                depth -= 1
                if depth == 0:
                    return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids.shape[1] <= self.prompt_len:
            return False
        new_tokens = input_ids[0, self.prompt_len :]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self.has_complete_json_object(decoded)


class AbstractSLMBackend(ABC):
    """Abstract interface for text-generation backends."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 160,
    ) -> Tuple[str, float, int]:
        """Generate response text and return (raw_text, latency_ms, tokens_generated)."""

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON dictionary from an LLM completion."""
        if not text:
            return None

        cleaned = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        candidates: List[str] = []
        if fence_match:
            candidates.append(fence_match.group(1))

        # String-literal and escape-aware balanced-brace scanner matching FirstJsonObjectStoppingCriteria.
        repaired_candidates: List[str] = []
        for idx, ch in enumerate(cleaned):
            if ch != "{":
                continue
            depth = 0
            in_string = False
            escaped = False
            balanced = False
            for end_idx in range(idx, len(cleaned)):
                c = cleaned[end_idx]
                if escaped:
                    escaped = False
                    continue
                if c == "\\" and in_string:
                    escaped = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(cleaned[idx : end_idx + 1])
                        balanced = True
                        break
            if not balanced and not in_string and 1 <= depth <= 3:
                repaired_candidates.append(cleaned[idx:] + ("}" * depth))

        fallback_dict: Optional[Dict[str, Any]] = None
        for cand in candidates + repaired_candidates:
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    if "action" in parsed:
                        return parsed
                    if fallback_dict is None:
                        fallback_dict = parsed
            except json.JSONDecodeError:
                continue
        return fallback_dict


class HuggingFaceSLMBackend(AbstractSLMBackend):
    """Local GPU inference backend with static prefix KV-caching and JSON brace early stopping."""

    _SPLIT_MARKER = "__SLM_STATE_BOUNDARY__"

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        logger.info("Loading HuggingFace model %s (quantization=%s)", self.model_name, config.quantization)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"device_map": "auto"}
        if config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()
        # Static system-prefix KV cache: {prefix_str: (prefix_ids, prefix_kv)}
        self._prefix_kv_cache: Dict[str, Tuple[torch.Tensor, Any]] = {}
        self._kv_cache_supported: bool = True

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

    def _get_or_build_prefix_kv(
        self,
        messages: List[Dict[str, str]],
        full_prompt: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Any]]:
        """
        If `messages` is a standard [system, user] turn, cache the tokenized & KV-encoded
        static system prefix so each step only prefills the dynamic user/state tokens.
        """
        if not self._kv_cache_supported or len(messages) != 2 or messages[0].get("role") != "system":
            return None

        probe_msgs = [messages[0], {"role": "user", "content": self._SPLIT_MARKER}]
        probe_text = self._render_prompt(probe_msgs)
        if self._SPLIT_MARKER not in probe_text:
            return None

        prefix_str, suffix_tail = probe_text.split(self._SPLIT_MARKER, 1)
        expected_suffix = messages[1].get("content", "") + suffix_tail
        if full_prompt != prefix_str + expected_suffix:
            return None

        if prefix_str not in self._prefix_kv_cache:
            prefix_ids = self.tokenizer(
                prefix_str, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.model.device)
            if prefix_ids.shape[1] < 8:
                return None
            with torch.inference_mode():
                out = self.model(input_ids=prefix_ids, use_cache=True)
            self._prefix_kv_cache[prefix_str] = (prefix_ids, out.past_key_values)

        prefix_ids, prefix_kv = self._prefix_kv_cache[prefix_str]
        suffix_ids = self.tokenizer(
            expected_suffix, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.model.device)
        full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
        return full_ids, torch.ones_like(full_ids), copy.deepcopy(prefix_kv)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 160,
    ) -> Tuple[str, float, int]:
        start_t = time.perf_counter()
        prompt_text = self._render_prompt(messages)

        kv_bundle = self._get_or_build_prefix_kv(messages, prompt_text)
        if kv_bundle is not None:
            input_ids, attention_mask, past_kv = kv_bundle
        else:
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            past_kv = None

        input_len = int(input_ids.shape[1])
        stopping = StoppingCriteriaList([FirstJsonObjectStoppingCriteria(self.tokenizer, input_len)])

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping,
            "use_cache": True,
        }
        if past_kv is not None:
            gen_kwargs["past_key_values"] = past_kv

        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        try:
            with torch.inference_mode():
                outputs = self.model.generate(**gen_kwargs)
        except Exception as exc:
            if past_kv is not None:
                logger.warning("Prefix KV-cache disabled after fallback error: %s", exc)
                self._kv_cache_supported = False
                gen_kwargs.pop("past_key_values", None)
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                gen_kwargs["input_ids"] = inputs["input_ids"]
                gen_kwargs["attention_mask"] = inputs["attention_mask"]
                input_len = int(inputs["input_ids"].shape[1])
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [FirstJsonObjectStoppingCriteria(self.tokenizer, input_len)]
                )
                with torch.inference_mode():
                    outputs = self.model.generate(**gen_kwargs)
            else:
                raise

        new_tokens = outputs[0][input_len:]
        num_tokens = int(new_tokens.shape[0])
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        return raw_text, latency_ms, num_tokens


class MockSLMBackend(AbstractSLMBackend):
    """
    Domain-agnostic deterministic mock backend for unit testing.
    Parses candidates dynamically from the prompt without any hardcoded IPs/services.
    """

    def __init__(self, scripted_responses: Optional[List[str]] = None) -> None:
        self.scripted_responses = list(scripted_responses) if scripted_responses else []
        self._call_idx = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 160,
    ) -> Tuple[str, float, int]:
        if self._call_idx < len(self.scripted_responses):
            resp = self.scripted_responses[self._call_idx]
            self._call_idx += 1
            return resp, 1.5, len(resp.split())

        prompt = messages[-1]["content"] if messages else ""
        cand_match = re.search(
            r"\d+\.\s+action=(\w+)\s+\|\s+parameters=(\{.*?\})\s+\|",
            prompt,
        )
        if cand_match:
            action_name = cand_match.group(1)
            params = ast.literal_eval(cand_match.group(2))
            payload = {"action": action_name, "parameters": params}
            raw = json.dumps(payload)
            return raw, 1.0, len(raw.split())

        cidr_match = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\b", prompt)
        host_matches = re.findall(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", prompt)
        if cidr_match and host_matches:
            payload = {
                "action": "ScanNetwork",
                "parameters": {"target_network": cidr_match.group(1), "source_host": host_matches[0]},
            }
            raw = json.dumps(payload)
            return raw, 1.0, len(raw.split())

        return '{"action": "ScanNetwork", "parameters": {}}', 1.0, 6
