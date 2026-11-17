"""
agent_wrapper.py — Domain-Agnostic SLM Attacker Wrapper with 1-Step Counterfactual Reflexion.

Integrates SLM backends (`slm_backend.py`) and Topological Belief Graph + Multi-Scale
Minimax Hazard Utility (`memory_module.py`) with NetSecGame while preserving genuine SLM autonomy:
- Tracks FirstPassValid, PostReflexionValid, and Wrapper Override rates explicitly.
- When `enable_action_masking=True` and the first output is invalid or redundant, grants
  the SLM one corrective turn before falling back to the top minimax utility action.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from netsecgame import Action, AgentRole, BaseAgent, GameState, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state

from .memory_module import (
    EpisodicMemoryBuffer,
    SemanticMemoryStore,
    canonicalize_action_type,
    canonicalize_parameters,
)
from .slm_backend import AbstractSLMBackend, SLMConfig

logger = logging.getLogger(__name__)


class BaseModularAgent:
    """Thin NetSecGame TCP client wrapper cleanly separating transport from planning."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, role: AgentRole = AgentRole.Attacker) -> None:
        self.host = host
        self.port = port
        self.role = role
        self.network_agent = BaseAgent(host, port, role)
        self._rng = random.Random(42)

    @property
    def _socket(self) -> Any:
        return getattr(self.network_agent, "_socket", None)

    def connect(self) -> Optional[Observation]:
        """Register this agent with the running NetSecGame server."""
        return self.network_agent.register()

    def reset_environment(self, randomize_topology: bool = True, seed: Optional[int] = None) -> Observation:
        """Reset the NetSecGame episode with a non-None integer seed when randomizing."""
        effective_seed = seed if seed is not None else self._rng.randint(1, 2**31 - 1)
        return self.network_agent.request_game_reset(
            request_trajectory=False,
            randomize_topology=randomize_topology,
            seed=effective_seed,
        )

    def step(self, action: Action) -> Observation:
        """Submit an Action object to NetSecGame and return the resulting Observation."""
        return self.network_agent.make_step(action)

    def terminate(self) -> None:
        """Close connection to the NetSecGame server cleanly."""
        self.network_agent.terminate_connection()


class ModernSLMAttacker(BaseModularAgent):
    """
    Domain-agnostic Stealth-Aware Topological Belief Graph SLM Attacker
    with Multi-Scale Minimax Hazard Utility and 1-Step Counterfactual Reflexion.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an autonomous penetration testing agent operating inside an authorized "
        "cybersecurity simulation environment (NetSecGame).\n"
        "You receive structured observations describing known networks, hosts, services, data, "
        "a Topological Belief Graph (UNEXPLORED / MAPPED / COMPROMISED / EXHAUSTED), a Temporal "
        "Stealth Advisory, and a set of permissible unexplored actions.\n"
        "Your objective is to complete the stated mission goal (such as discovering target data "
        "and exfiltrating it to the designated host) in minimal steps while avoiding SIEM "
        "detection.\n"
        "Valid NetSecGame action types: ScanNetwork, FindServices, ExploitService, FindData, "
        "ExfiltrateData.\n"
        "Respond strictly with ONE valid JSON object containing keys \"action\" and \"parameters\". "
        "Do not output prose outside the JSON object."
    )

    REFLEXION_CORRECTION_MSG = (
        "Action rejected: already executed or invalid parameters. "
        "Select strictly from the listed permissible actions and output ONE valid JSON object."
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
        Validate unmodified JSON output against NetSecGame rules and semantic history.
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
        Domain-agnostic fallback selecting the top Multi-Scale Minimax Hazard Utility U(a)
        candidate from the Belief Graph when action masking is enabled.
        """
        candidates = self.semantic_memory.get_valid_action_space(
            state, include_executed=False, episodic_memory=self.episodic_memory
        )
        if not candidates:
            candidates = self.semantic_memory.get_valid_action_space(
                state, include_executed=True, episodic_memory=self.episodic_memory
            )
        if not candidates:
            return None, {}

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for cand in candidates:
            utility, _, _ = self.semantic_memory.compute_minimax_hazard_utility(
                cand["action"],
                cand["parameters"],
                episodic_memory=self.episodic_memory,
            )
            scored.append((utility, cand))

        max_u = max(u for u, _ in scored)
        top_pool = [c for u, c in scored if abs(u - max_u) <= 1e-6]
        chosen_dict = self._fallback_rng.choice(top_pool)

        try:
            valid, netsec_action = create_action_from_response(chosen_dict, state)
        except (KeyError, TypeError, ValueError, AttributeError):
            valid, netsec_action = False, None
        if valid and netsec_action is not None:
            return netsec_action, chosen_dict

        for _, cand in sorted(scored, key=lambda x: -x[0]):
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
        Generate the next NetSecGame Action with optional 1-step counterfactual reflexion
        and populate `self.last_decision_metadata`.
        """
        state = observation.state
        if not goal_description:
            info = getattr(observation, "info", None)
            if isinstance(info, dict):
                goal_description = str(info.get("goal_description", "") or "")

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
        first_pass_raw_valid = bool(raw_valid_syntax and raw_valid_in_state)
        first_pass_valid = bool(
            first_pass_raw_valid and not (self.config.use_memory and raw_redundant)
        )
        post_reflexion_valid = first_pass_valid
        reflexion_triggered = False

        final_action = raw_action
        final_dict = raw_norm
        was_overridden = False

        if self.config.enable_action_masking:
            if not first_pass_valid:
                # 1-Step Counterfactual Reflexion before falling back to U(a)
                reflexion_triggered = True
                reflexion_messages = list(messages) + [
                    {"role": "assistant", "content": raw_text or "{}"},
                    {"role": "user", "content": self.REFLEXION_CORRECTION_MSG},
                ]
                r_text, r_latency, r_tokens = self.backend.generate(
                    reflexion_messages,
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens,
                )
                latency_ms += r_latency
                tokens += r_tokens
                r_parsed = AbstractSLMBackend.extract_json(r_text)
                r_syn, r_state, r_red, r_act, r_norm = self._validate_raw_decision(r_parsed, state)
                r_ok = bool(r_syn and r_state and not (self.config.use_memory and r_red))
                if r_ok and r_act is not None:
                    post_reflexion_valid = True
                    final_action = r_act
                    final_dict = r_norm
                else:
                    fallback_action, fallback_dict = self.select_masked_fallback(state)
                    if fallback_action is not None:
                        final_action = fallback_action
                        final_dict = fallback_dict
                        was_overridden = True
                    elif not first_pass_raw_valid:
                        final_action = None
        else:
            if not first_pass_raw_valid:
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
            "raw_slm_valid": bool(raw_valid_syntax and raw_valid_in_state),
            "first_pass_valid": first_pass_valid,
            "post_reflexion_valid": post_reflexion_valid,
            "reflexion_triggered": reflexion_triggered,
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
