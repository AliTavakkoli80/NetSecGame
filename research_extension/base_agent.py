"""
base_agent.py — Backwards-compatible re-export of SLM backends and agent wrapper.
"""

from .agent_wrapper import BaseModularAgent, ModernSLMAttacker
from .slm_backend import (
    AbstractSLMBackend,
    FirstJsonObjectStoppingCriteria,
    HuggingFaceSLMBackend,
    MockSLMBackend,
    SLMConfig,
)

__all__ = [
    "AbstractSLMBackend",
    "BaseModularAgent",
    "FirstJsonObjectStoppingCriteria",
    "HuggingFaceSLMBackend",
    "MockSLMBackend",
    "ModernSLMAttacker",
    "SLMConfig",
]
