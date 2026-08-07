"""GRPO fine-tuning for cryptanalysis, with pinned deterministic verifiers.

``tasks``, ``verifiers``, ``rewards`` and ``dataset`` are standard library only
and import without torch. The trainer and the evaluator are the only modules
that need a GPU stack, so the reward pipeline stays testable in CI.
"""

from .tasks import FAMILIES, Task, generate
from .verifiers import Verdict, verify, verify_task

__all__ = ["FAMILIES", "Task", "Verdict", "generate", "verify", "verify_task"]
