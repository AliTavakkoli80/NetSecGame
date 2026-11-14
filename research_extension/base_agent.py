"""
base_agent.py — Modular Domain-Agnostic SLM Attacker & NetSecGame Bridge.

Integrates HuggingFace quantized/bfloat16 Small Language Models (SLMs) and mock backends
with the NetSecGame environment while preserving genuine LLM autonomy:
- Raw SLM outputs are strictly validated before any intervention.
- When enable_action_masking=False (raw ReAct baseline), NO fallback override occurs.
- When enable_action_masking=True, intervention triggers ONLY if the SLM produces
  syntactically invalid JSON, an illegal NetSecGame action, or an exact repeat
  already recorded in semantic memory—and every intervention is logged explicitly.
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from netsecgame import Action, ActionType, AgentRole, BaseAgent, GameState, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state

from .memory_module import (
    EpisodicMemoryBuffer,
    SemanticMemoryStore,
    canonicalize_action_type,
    canonicalize_parameters,
)

logger = logging.getLogger(__name__)


@dataclass
class SLMConfig:
    """Configuration for Small Language Model inference and agent modularity."""

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    backend: str = "huggingface"  # "huggingface" or "mock"
    quantization: Optional[str] = None  # None ("bfloat16") or "4bit"
    temperature: float = 0.2
    max_new_tokens: int = 256
    use_memory: bool = True
    enable_action_masking: bool = True
    enable_stealth_pacing: bool = True
    memory_window: int = 5
    device: str = "cuda"


class AbstractSLMBackend(ABC):
    """Abstract interface for text-generation backends."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float, int]:
        """Generate response text and return (raw_text, latency_ms, tokens_generated)."""

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON dictionary from an LLM completion."""
        if not text:
            return None

        cleaned = text.strip()
        # Strip optional markdown code fences first.
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        candidates: List[str] = []
        if fence_match:
            candidates.append(fence_match.group(1))

        # Balance braces across all '{' positions in cleaned text as fallback.
        for idx, ch in enumerate(cleaned):
            if ch == "{":
                depth = 0
                for end_idx in range(idx, len(cleaned)):
                    c = cleaned[end_idx]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append(cleaned[idx : end_idx + 1])
                            break

        for cand in candidates:
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None


class HuggingFaceSLMBackend(AbstractSLMBackend):
    """Local GPU inference backend using Transformers (supports bfloat16 & 4-bit NF4)."""

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

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float, int]:
        start_t = time.perf_counter()

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

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
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float, int]:
        if self._call_idx < len(self.scripted_responses):
            resp = self.scripted_responses[self._call_idx]
            self._call_idx += 1
            return resp, 1.5, len(resp.split())

        prompt = messages[-1]["content"] if messages else ""
        # 1. If Permissible Unexplored Actions section is present, select first parsed entry.
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

        # 2. Raw-state fallback: extract network/host strings dynamically from observation text.
        cidr_match = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\b", prompt)
        host_matches = re.findall(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", prompt)
        if cidr_match and host_matches:
            payload = {
                "action": "ScanNetwork",
                "parameters": {"target_network": cidr_match.group(1), "source_host": host_matches[0]},
            }
            raw = json.dumps(payload)
            return raw, 1.0, len(raw.split())

        return '{"action": "NoOp", "parameters": {}}', 1.0, 4


class BaseModularAgent:
    """Thin NetSecGame connection wrapper supporting deterministic or randomized resets."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, role: AgentRole = AgentRole.Attacker) -> None:
        self.host = host
        self.port = port
        self.role = role
        self.network_agent = BaseAgent(host, port, role)
        self._rng = random.SystemRandom()

    def connect(self) -> Observation:
        """Register with NetSecGame server and return initial observation."""
        obs = self.network_agent.register()
        return obs

    def reset_environment(
        self,
        randomize_topology: bool = True,
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Reset the NetSecGame episode.
        NetSecGame requires a non-None integer seed whenever randomize_topology=True.
        """
        effective_seed = seed if seed is not None else self._rng.randint(1, 2**31 - 1)
        obs = self.network_agent.request_game_reset(
            request_trajectory=False,
            randomize_topology=randomize_topology,
            seed=effective_seed,
        )
        return obs

    def step(self, action: Action) -> Observation:
        """Submit an Action object to NetSecGame and return the resulting Observation."""
        return self.network_agent.make_step(action)

    def terminate(self) -> None:
        """Close connection to the NetSecGame server cleanly."""
        self.network_agent.terminate_connection()


class ModernSLMAttacker(BaseModularAgent):
    """
    Domain-agnostic Stealth-Aware Topological Belief Graph SLM Attacker.
    Supports ablation toggles (`use_memory`, `enable_action_masking`) cleanly and
    logs raw SLM autonomy vs wrapper interventions transparently.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an autonomous penetration testing agent operating inside an authorized "
        "cybersecurity simulation environment (NetSecGame).\n"
        "You receive structured observations describing known networks, hosts, services, data, "
        "a Topological Belief Graph (UNEXPLORED / MAPPED / COMPROMISED / EXHAUSTED), a Temporal "
        "Stealth Advisory over the recent 5-action sliding window, and a set of permissible "
        "unexplored actions.\n"
        "Your objective is to complete the stated mission goal (such as discovering target data "
        "and exfiltrating it to the designated host) in minimal steps while avoiding SIEM "
        "detection.\n"
        "Valid NetSecGame action types: ScanNetwork, FindServices, ExploitService, FindData, "
        "ExfiltrateData.\n"
        "Respond strictly with ONE valid JSON object containing keys \"action\" and \"parameters\". "
        "Do not output prose outside the JSON object."
    )

    def __init__(
        self,
        config: SLMConfig,
        backend: AbstractSLMBackend,
        host: str = "127.0.0.1",
        port: int = 9000,
        enable_stealth_pacing: Optional[bool] = None,
    ) -> None:
        super().__init__(host=host, port=port, role=AgentRole.Attacker)
        self.config = config
        self.backend = backend
        self.enable_stealth_pacing = (
            config.enable_stealth_pacing if enable_stealth_pacing is None else bool(enable_stealth_pacing)
        )
        self.episodic_memory = EpisodicMemoryBuffer(
            max_capacity=max(config.memory_window * 2, 10),
            stealth_window=config.memory_window,
        )
        self.semantic_memory = SemanticMemoryStore(
            stealth_window=config.memory_window,
            enable_stealth_pacing=self.enable_stealth_pacing,
        )
        self._fallback_rng = random.Random(42)
        self.last_decision_metadata: Dict[str, Any] = {}

    def reset_memory(self) -> None:
        """Clear episodic and semantic state at the start of a new episode."""
        self.episodic_memory.clear()
        self.semantic_memory.reset_episode()
        self.last_decision_metadata = {}

    def _build_raw_react_prompt(self, state: GameState, goal_description: str) -> str:
        """Construct a pure memoryless ReAct-style observation prompt (Config B baseline)."""
        status_text = create_status_from_state(state)
        goal_line = goal_description or "Discover target data and exfiltrate it to the goal host."
        return (
            f"Mission Goal: {goal_line}\n\n"
            f"Current Observation:\n{status_text}\n\n"
            "Available actions and parameter schemas:\n"
            "- ScanNetwork: {\"target_network\": \"<cidr>\", \"source_host\": \"<ip>\"}\n"
            "- FindServices: {\"target_host\": \"<ip>\", \"source_host\": \"<ip>\"}\n"
            "- ExploitService: {\"target_host\": \"<ip>\", \"target_service\": \"<service>\", \"source_host\": \"<ip>\"}\n"
            "- FindData: {\"target_host\": \"<ip>\", \"source_host\": \"<ip>\"}\n"
            "- ExfiltrateData: {\"target_host\": \"<ip>\", \"source_host\": \"<ip>\", \"data\": {\"owner\": \"<owner>\", \"id\": \"<id>\"}}\n\n"
            "Output ONLY a single JSON object with keys \"action\" and \"parameters\"."
        )

    def _validate_raw_decision(
        self,
        parsed_json: Optional[Dict[str, Any]],
        state: GameState,
    ) -> Tuple[bool, bool, bool, Optional[Action], Dict[str, Any]]:
        """
        Validate the SLM's unmodified JSON output against NetSecGame rules and semantic history.
        Returns: (valid_syntax, valid_in_state, is_redundant, netsec_action, normalized_dict)
        """
        if not parsed_json or not isinstance(parsed_json, dict) or "action" not in parsed_json:
            return False, False, False, None, {}

        raw_params = parsed_json.get("parameters", {})
        if not isinstance(raw_params, dict):
            return False, False, False, None, {}

        canon_action = canonicalize_action_type(parsed_json["action"])
        canon_params = canonicalize_parameters(raw_params)
        normalized = {"action": canon_action, "parameters": canon_params}

        try:
            is_state_valid, netsec_action = create_action_from_response(normalized, state)
        except (KeyError, TypeError, ValueError, AttributeError):
            return True, False, False, None, normalized

        if not is_state_valid or netsec_action is None:
            return True, False, False, None, normalized

        is_redundant = self.semantic_memory.is_action_redundant(canon_action, canon_params)
        return True, True, is_redundant, netsec_action, normalized

    def select_masked_fallback(self, state: GameState) -> Tuple[Optional[Action], Dict[str, Any]]:
        """
        Domain-agnostic fallback when action masking is enabled and raw SLM output failed
        validation or repeated an already executed action.
        Selects among valid unexplored actions minimizing SIEM stealth risk without any
        IP, service, data, or hardcoded phase ordering.
        """
        candidates = self.semantic_memory.get_valid_action_space(
            state, include_executed=False, episodic_memory=self.episodic_memory
        )
        if not candidates:
            # All exact pairs exhausted; allow non-executed or valid reset options from executed space.
            candidates = self.semantic_memory.get_valid_action_space(
                state, include_executed=True, episodic_memory=self.episodic_memory
            )

        if not candidates:
            return None, {}

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for cand in candidates:
            risk = self.semantic_memory.score_action_stealth_risk(
                cand["action"],
                cand["parameters"],
                episodic_memory=self.episodic_memory,
            )
            scored.append((risk, cand))

        min_risk = min(r for r, _ in scored)
        lowest_risk_pool = [c for r, c in scored if abs(r - min_risk) <= 1e-6]
        chosen_dict = self._fallback_rng.choice(lowest_risk_pool)

        try:
            valid, netsec_action = create_action_from_response(chosen_dict, state)
        except (KeyError, TypeError, ValueError, AttributeError):
            valid, netsec_action = False, None
        if valid and netsec_action is not None:
            return netsec_action, chosen_dict

        # Linearly scan remaining scored candidates if edge validation rejected one.
        for _, cand in sorted(scored, key=lambda x: x[0]):
            try:
                ok, act = create_action_from_response(cand, state)
            except (KeyError, TypeError, ValueError, AttributeError):
                ok, act = False, None
            if ok and act is not None:
                return act, cand
        return None, {}

    def plan_next_action(
        self,
        observation: Observation,
        goal_description: str = "",
    ) -> Tuple[Optional[Action], Dict[str, Any], float, int]:
        """
        Generate the next NetSecGame Action for `observation` and populate
        `self.last_decision_metadata` with explicit raw autonomy vs override telemetry.
        """
        state = observation.state
        if not goal_description:
            info = getattr(observation, "info", None)
            if isinstance(info, dict):
                goal_description = str(info.get("goal_description", "") or "")

        # Always keep stealth/state belief synced for telemetry, while gating prompt context on use_memory.
        self.semantic_memory.update_from_observation(observation, goal_description=goal_description)

        if self.config.use_memory:
            user_prompt = self.semantic_memory.format_context_prompt(
                observation,
                episodic_memory=self.episodic_memory,
                goal_description=goal_description,
            )
            stealth_deferred_count = self.semantic_memory.last_stealth_deferred_count
        else:
            user_prompt = self._build_raw_react_prompt(state, goal_description)
            stealth_deferred_count = 0

        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_text, latency_ms, tokens = self.backend.generate(
            messages,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )
        parsed_json = AbstractSLMBackend.extract_json(raw_text)

        raw_valid_syntax, raw_valid_in_state, raw_redundant, raw_action, raw_norm = (
            self._validate_raw_decision(parsed_json, state)
        )
        raw_slm_valid = bool(raw_valid_syntax and raw_valid_in_state)

        final_action = raw_action
        final_dict = raw_norm
        was_overridden = False

        if self.config.enable_action_masking:
            needs_recovery = (not raw_slm_valid) or (self.config.use_memory and raw_redundant)
            if needs_recovery:
                fallback_action, fallback_dict = self.select_masked_fallback(state)
                if fallback_action is not None:
                    final_action = fallback_action
                    final_dict = fallback_dict
                    was_overridden = True
                elif not raw_slm_valid:
                    final_action = None
        else:
            # Strict no-masking mode (e.g., Config B): never substitute a heuristic fallback.
            if not raw_slm_valid:
                final_action = None

        executed_type = final_dict.get("action", "INVALID")
        executed_params = final_dict.get("parameters", {})
        stealth_risk = (
            self.semantic_memory.score_action_stealth_risk(
                executed_type,
                executed_params,
                episodic_memory=self.episodic_memory,
            )
            if executed_type != "INVALID"
            else 0.0
        )

        self.last_decision_metadata = {
            "raw_response": raw_text,
            "raw_valid_syntax": raw_valid_syntax,
            "raw_valid_in_state": raw_valid_in_state,
            "raw_slm_valid": raw_slm_valid,
            "raw_redundant": raw_redundant,
            "was_overridden": was_overridden,
            "stealth_risk_score": stealth_risk,
            "stealth_deferred_count": int(stealth_deferred_count),
            "latency_ms": latency_ms,
            "tokens_generated": tokens,
            "raw_action": raw_norm,
            "chosen_action": final_dict,
        }

        return final_action, final_dict, latency_ms, tokens
