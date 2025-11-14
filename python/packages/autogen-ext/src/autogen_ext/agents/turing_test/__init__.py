"""
AutoGen Turing Test Extension

This module provides agents and orchestration for conducting Turing Test experiments
within the AutoGen framework. It enables systematic evaluation of AI agents through
human-AI discrimination tasks with temperature matrix testing.

Authors: Jacob Choi (Colby College), Mark Encarnación (Microsoft Research)
"""

from ._evaluator_agent import EvaluatorAgent
from ._interrogator_agent import InterrogatorAgent
from ._subject_agent import SubjectAgent

__all__ = ["InterrogatorAgent", "SubjectAgent", "EvaluatorAgent"]

__version__ = "0.1.0"
