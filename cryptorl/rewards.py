"""GRPO reward functions.

TRL sums the reward functions it is given, so each one here returns a list of
floats already multiplied by its weight. The weights are the actual design
decision in this file, and the ordering constraint they encode is:

    a correct answer must out-earn every incorrect one, whatever else it does

If partial credit for a near-miss can ever exceed the correctness bonus, GRPO
will find that out faster than you will. ``RewardWeights.check()`` asserts the
ordering rather than leaving it to whoever edits the config next.

The other thing this file exists to prevent is shotgunning: a model that emits
ten candidate answers and lets the grader find the good one is not doing
cryptanalysis, it is exploiting a lenient parser. The parser takes the first
JSON object after ``</think>`` and refuses outright if there is more than one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .verifiers import Verdict, verify

_THINK_CLOSE = "</think>"
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class Parsed:
    think: str
    answer_region: str
    artifact: object | None
    fenced: bool
    violations: tuple[str, ...]

    @property
    def well_formed(self) -> bool:
        return self.artifact is not None and not self.violations


def parse_completion(completion: object) -> Parsed:
    """Split a rollout into its reasoning and its graded artifact.

    Qwen3's chat template opens the thinking block itself when
    ``enable_thinking=True``, so a completion legitimately starts *inside*
    ``<think>`` and carries only the closing tag. Requiring the opening tag here
    would zero the format reward on every rollout and teach the model to emit a
    second, nested ``<think>``.
    """
    text = _completion_text(completion)
    violations: list[str] = []

    if text.count(_THINK_CLOSE) > 1:
        violations.append("multiple_think_blocks")
    if _THINK_CLOSE in text:
        head, _, tail = text.partition(_THINK_CLOSE)
        think = head.replace("<think>", "").strip()
        answer_region = tail
        if not think:
            violations.append("empty_think")
    else:
        # No answer region at all, rather than "the whole thing".
        #
        # The contract is that the answer follows </think>; with no closing tag
        # the model never left its scratchpad. Treating the reasoning as the
        # answer region meant the illustrative JSON a model writes while
        # thinking ("suppose the answer were {...}") counted as candidate
        # answers, so a run that had simply failed to terminate was reported as
        # `multiple_answers` -- an anti-shotgunning rule firing on something
        # that was not shotgunning. The reward is the same either way; the
        # difference is whether the reason tells you what actually went wrong.
        violations.append("unterminated_think")
        think, answer_region = text, ""

    fences = _FENCE_RE.findall(answer_region)
    if len(fences) > 1:
        violations.append("multiple_answers")
        return Parsed(think, answer_region, None, True, tuple(violations))

    fenced = bool(fences)
    if fenced:
        candidate = fences[0]
    else:
        violations.append("unfenced_answer")
        bare = _BARE_OBJECT_RE.findall(answer_region)
        if len(bare) > 1:
            violations.append("multiple_answers")
            return Parsed(think, answer_region, None, False, tuple(violations))
        candidate = bare[0] if bare else ""

    if not candidate:
        violations.append("no_answer")
        return Parsed(think, answer_region, None, fenced, tuple(violations))

    try:
        artifact = json.loads(candidate)
    except json.JSONDecodeError:
        violations.append("unparseable_json")
        artifact = None

    return Parsed(think, answer_region, artifact, fenced, tuple(violations))


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Conversational datasets hand back a message list.
        return "".join(str(m.get("content", "")) for m in completion if isinstance(m, dict))
    return str(completion)


@dataclass(frozen=True)
class RewardWeights:
    correctness: float = 2.0
    partial: float = 0.4
    format: float = 0.3
    invalid_penalty: float = 0.6
    budget_penalty: float = 0.2
    think_budget_chars: int = 6000

    def check(self) -> None:
        best_wrong = self.partial + self.format
        worst_right = self.correctness
        if best_wrong >= worst_right:
            raise ValueError(
                f"a perfectly formatted near-miss can score {best_wrong} while a "
                f"correct answer is worth {worst_right}; lower `partial`/`format` "
                "or raise `correctness`"
            )


def grade(completion: object, family: str, solution: dict, params: dict) -> tuple[Parsed, Verdict]:
    parsed = parse_completion(completion)
    # Checked before the null test, not after: shotgunning already zeroes the
    # artifact, and reporting it as "nothing to grade" would hide the one
    # failure mode whose whole point is that it looks like a good answer.
    if "multiple_answers" in parsed.violations:
        return parsed, Verdict(False, "more than one candidate answer", invalid=True)
    if parsed.artifact is None:
        return parsed, Verdict(False, "no gradeable artifact", invalid=True)
    return parsed, verify(family, parsed.artifact, solution, params)


def _rows(kwargs: dict, n: int) -> list[tuple[str, dict, dict]]:
    families = kwargs["family"]
    solutions = kwargs["solution_json"]
    params = kwargs["params_json"]
    return [
        (families[i], json.loads(solutions[i]), json.loads(params[i])) for i in range(n)
    ]


def build_reward_functions(weights: RewardWeights | None = None) -> list:
    """Reward callables in TRL's shape: ``f(completions, **columns) -> list[float]``.

    Each one re-grades the batch instead of sharing a memo. Verification is pure
    integer arithmetic on a rollout that cost thousands of forward passes to
    produce, so the memo would save nothing measurable while introducing a cache
    whose staleness would show up as silently wrong rewards.
    """
    weights = weights or RewardWeights()
    weights.check()

    def graded(completions, kwargs) -> list[tuple[Parsed, Verdict]]:
        rows = _rows(kwargs, len(completions))
        return [grade(c, *row) for c, row in zip(completions, rows)]

    def correctness_reward(completions, **kwargs):
        return [weights.correctness if v.accepted else 0.0 for _, v in graded(completions, kwargs)]

    def partial_credit_reward(completions, **kwargs):
        return [weights.partial * v.partial for _, v in graded(completions, kwargs)]

    def format_reward(completions, **kwargs):
        scores = []
        for parsed, _ in graded(completions, kwargs):
            if parsed.well_formed:
                score = 1.0
            elif parsed.artifact is not None:
                score = 0.5  # gradeable but sloppy: unfenced, or no thinking block
            else:
                score = 0.0
            scores.append(weights.format * score)
        return scores

    def invalid_step_penalty(completions, **kwargs):
        return [
            -weights.invalid_penalty if v.invalid else 0.0
            for _, v in graded(completions, kwargs)
        ]

    def thinking_budget_penalty(completions, **kwargs):
        budget = weights.think_budget_chars
        scores = []
        for parsed, _ in graded(completions, kwargs):
            overrun = max(0, len(parsed.think) - budget)
            scores.append(-weights.budget_penalty * min(1.0, overrun / budget))
        return scores

    return [
        correctness_reward,
        partial_credit_reward,
        format_reward,
        invalid_step_penalty,
        thinking_budget_penalty,
    ]
