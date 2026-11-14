"""
Research Extension for Cyber-SLM Agent.

Provides modernized Hugging Face local SLM pipelines, episodic/semantic memory,
and comprehensive benchmark instrumentation for NetSecGame.
"""

from research_extension.base_agent import (
    SLMConfig,
    AbstractSLMBackend,
    HuggingFaceSLMBackend,
    MockSLMBackend,
    BaseModularAgent,
    ModernSLMAttacker,
)
from research_extension.memory_module import (
    StepMemory,
    EpisodicMemoryBuffer,
    SemanticMemoryStore,
)
from research_extension.metrics_logger import (
    StepMetric,
    EpisodeMetric,
    MetricsLogger,
)
from research_extension.benchmark_runner import BenchmarkHarness

__all__ = [
    "SLMConfig",
    "AbstractSLMBackend",
    "HuggingFaceSLMBackend",
    "MockSLMBackend",
    "BaseModularAgent",
    "ModernSLMAttacker",
    "StepMemory",
    "EpisodicMemoryBuffer",
    "SemanticMemoryStore",
    "StepMetric",
    "EpisodeMetric",
    "MetricsLogger",
    "BenchmarkHarness",
]
