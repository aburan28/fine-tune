"""Tests for the parts of the pipeline that decide what the model is paid for.

The trainer is not tested here -- it needs a GPU and it is mostly configuration.
The generators, the verifiers and the reward arithmetic are, because a bug in
any of them does not crash: it trains a model to do the wrong thing, and you
find out a week later from an evaluation that looks fine.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

from cryptorl import primitives as prim
from cryptorl.dataset import EVAL_SEED_BASE, build_splits
from cryptorl.rewards import (
    RewardWeights,
    build_reward_functions,
    grade,
    parse_completion,
)
from cryptorl.tasks import FAMILIES, MAX_DIFFICULTY, generate
from cryptorl.verifiers import verify, verify_task

ALL_INSTANCES = [(f, d) for f in FAMILIES for d in range(MAX_DIFFICULTY + 1)]

# The `dlp-secp256k1` family mirrors a live objective on the proofwork network,
# whose checker is hashed into the objective's id:
# https://github.com/aburan28/distributed-researcher/blob/main/examples/ecdlp/checkers/secp256k1_dlog.py
#
# When that repository is checked out nearby, the parity test below runs against
# it. When it is not, the group-law tests still pin this curve arithmetic on
# their own -- deliberately, because vendoring a copy of a checker whose whole
# value is being pinned elsewhere would make the parity test compare a file to
# itself.
_CHECKER_RELATIVE = Path("examples/ecdlp/checkers/secp256k1_dlog.py")


def _find_pinned_checker() -> Path | None:
    roots = []
    if os.environ.get("PROOFWORK_ROOT"):
        roots.append(Path(os.environ["PROOFWORK_ROOT"]))
    here = Path(__file__).resolve()
    roots += [*here.parents[:4], *(p / "distributed-researcher" for p in here.parents[:4])]
    for root in roots:
        candidate = root / _CHECKER_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def _load_pinned_checker(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_dlog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requires_pinned_checker = pytest.mark.skipif(
    _find_pinned_checker() is None,
    reason="proofwork checkout not found; set PROOFWORK_ROOT to run the parity test",
)


# --- generators verify against themselves ------------------------------------


@pytest.mark.parametrize("family,difficulty", ALL_INSTANCES)
def test_ground_truth_is_accepted(family, difficulty):
    for seed in range(4):
        task = generate(family, difficulty, seed)
        verdict = verify_task(task, task.solution)
        assert verdict.accepted, f"{family}/{difficulty}/{seed}: {verdict.reason}"
        assert verdict.partial == 1.0


@pytest.mark.parametrize("family,difficulty", ALL_INSTANCES)
def test_generation_is_deterministic(family, difficulty):
    a = generate(family, difficulty, 11)
    b = generate(family, difficulty, 11)
    assert a.prompt == b.prompt
    assert a.solution == b.solution
    assert a.task_id == b.task_id
    assert generate(family, difficulty, 12).prompt != a.prompt


@pytest.mark.parametrize("family,difficulty", ALL_INSTANCES)
def test_prompt_never_contains_the_answer(family, difficulty):
    """A prompt that leaks its own solution trains nothing and evaluates as a win."""
    task = generate(family, difficulty, 3)
    for key, value in task.solution.items():
        # Short values are skipped: a Caesar shift of 5 "appears in" the phrase
        # "a number 1-25" and always will. Everything long enough for a
        # substring hit to mean something -- keys, alphabets, k, m, d, factors --
        # is still checked.
        if len(str(value)) >= 4:
            assert str(value) not in task.prompt, f"{family}: {key} appears in the prompt"


def test_families_do_not_share_a_random_stream():
    """Seed n must not draw correlated instances across families."""
    shifts = {generate(f, 0, 5).solution.get("key") for f in ("caesar",)}
    assert shifts  # sanity
    a = generate("dlp-modp", 0, 5).solution["k"]
    b = generate("rsa-cube", 0, 5).solution["m"]
    assert a != b


# --- parity with the network's pinned checker --------------------------------


def _on_curve(point) -> bool:
    x, y = point
    return (y * y - x * x * x - 7) % prim.SECP256K1_P == 0


def test_scalar_multiplication_obeys_the_group_law():
    """Pins the curve arithmetic without an external vector.

    Every check here is an identity that a wrong implementation cannot satisfy
    by accident: the order annihilates the generator, negation is reflection in
    the x-axis, double-and-add agrees with repeated addition, and the map is
    additive in the scalar. Hard-coding a k*G constant instead would only test
    that the constant was transcribed correctly.
    """
    assert prim.ec_mul(prim.SECP256K1_N) is None, "N*G must be the point at infinity"

    gx, gy = prim.SECP256K1_G
    assert prim.ec_mul(prim.SECP256K1_N - 1) == (gx, (prim.SECP256K1_P - gy) % prim.SECP256K1_P)

    running = None
    for k in range(1, 32):
        running = prim.ec_add(running, prim.SECP256K1_G)
        assert running == prim.ec_mul(k), f"double-and-add disagrees at k={k}"
        assert _on_curve(running)

    assert prim.ec_mul(1234567) == prim.ec_add(prim.ec_mul(1000000), prim.ec_mul(234567))
    assert _on_curve(prim.ec_mul(2**39 + 12345))


@requires_pinned_checker
def test_ec_mul_matches_the_pinned_checker():
    """The proofwork ECDLP objective is hashed over that checker's bytes. If the
    two implementations of secp256k1 ever disagree, this one is the wrong one."""
    pinned = _load_pinned_checker(_find_pinned_checker())
    for k in (1, 2, 3, 7, 2**39 + 12345, prim.SECP256K1_N - 1):
        assert prim.ec_mul(k) == pinned._mul(k, (pinned.GX, pinned.GY))


@requires_pinned_checker
def test_pinned_checker_rejects_the_same_shapes_we_do():
    pinned = _load_pinned_checker(_find_pinned_checker())
    task = generate("dlp-secp256k1", 0, 0)
    for artifact in ({"k": "12"}, {"k": True}, {}):
        assert not pinned.check(artifact)[0]
        assert not verify_task(task, artifact).accepted


# --- verifier semantics ------------------------------------------------------


def test_wiener_accepts_any_working_private_exponent():
    """d is unique only modulo lcm(p-1, q-1); rejecting an equivalent key that
    decrypts correctly would fail a submission that is in fact right."""
    # Chosen so that d + lcm is still below n, which is the canonical range the
    # verifier enforces; the point is that the *stored* d is not what is checked.
    p, q = 586111, 784039
    n, phi = p * q, (p - 1) * (q - 1)
    e = 65537
    d = prim.invmod(e, phi)
    lcm = phi // math.gcd(p - 1, q - 1)
    assert d + lcm < n
    params = {"n": n, "e": e}
    assert verify("rsa-wiener", {"d": d}, {}, params).accepted
    assert verify("rsa-wiener", {"d": d + lcm}, {}, params).accepted
    assert not verify("rsa-wiener", {"d": d + 1}, {}, params).accepted


def test_certificate_verifiers_ignore_the_stored_solution():
    """They must decide from the instance alone, the way a pinned network
    checker does -- otherwise the reward is not reproducible by anyone else."""
    for family in (
        "dlp-modp",
        "dlp-secp256k1",
        "rsa-factor",
        "rsa-cube",
        "rsa-common-modulus",
        "rsa-wiener",
    ):
        task = generate(family, 0, 1)
        assert verify(family, task.solution, {}, task.params).accepted


def test_echoing_the_ciphertext_earns_nothing():
    for family in ("caesar", "vigenere", "substitution", "transposition"):
        task = generate(family, 1, 2)
        verdict = verify_task(task, {"plaintext": task.params["ciphertext"]})
        assert not verdict.accepted
        assert verdict.partial == 0.0
        assert verdict.invalid


def test_partial_credit_rewards_a_near_miss_but_not_a_guess():
    task = generate("caesar", 0, 0)
    truth = task.solution["plaintext"]
    near = truth[:-4] + "ZZZZ"
    key = task.solution["key"]

    guess = verify_task(task, {"plaintext": "Z" * len(truth)}).partial
    text_only = verify_task(task, {"plaintext": near}).partial
    with_key = verify_task(task, {"plaintext": near, "key": key}).partial

    assert guess < 0.1
    assert 0.6 < text_only < with_key < 1.0


def test_wrong_type_is_invalid_not_merely_wrong():
    task = generate("dlp-modp", 0, 0)
    assert verify_task(task, {"k": "42"}).invalid
    assert not verify_task(task, {"k": 1}).invalid  # in range, just wrong


def test_out_of_range_scalar_is_refused():
    task = generate("dlp-secp256k1", 0, 0)
    assert not verify_task(task, {"k": task.solution["k"] + task.params["high"]}).accepted


# --- completion parsing ------------------------------------------------------


def test_parses_a_completion_whose_think_tag_the_template_opened():
    """Qwen3 templates emit `<think>` as part of the prompt, so the completion
    carries only the closing tag. This must not read as malformed."""
    parsed = parse_completion('reasoning here\n</think>\n\n```json\n{"k": 7}\n```')
    assert parsed.artifact == {"k": 7}
    assert parsed.violations == ()
    assert parsed.well_formed


def test_parses_an_explicit_opening_tag_too():
    parsed = parse_completion('<think>work</think>\n```json\n{"k": 7}\n```')
    assert parsed.artifact == {"k": 7}
    assert parsed.violations == ()


def test_several_candidate_answers_are_refused():
    completion = 'work\n</think>\n```json\n{"k": 1}\n```\nor maybe\n```json\n{"k": 2}\n```'
    parsed = parse_completion(completion)
    assert parsed.artifact is None
    assert "multiple_answers" in parsed.violations


def test_shotgunning_cannot_be_rescued_by_including_the_right_answer():
    task = generate("dlp-modp", 0, 0)
    k = task.solution["k"]
    completion = f'work\n</think>\n```json\n{{"k": 1}}\n```\n```json\n{{"k": {k}}}\n```'
    _, verdict = grade(completion, task.family, task.solution, task.params)
    assert not verdict.accepted
    assert verdict.invalid


def test_unfenced_json_still_grades_but_is_marked_sloppy():
    """Correctness has to be reachable before formatting is learned, or GRPO
    gets no gradient at all in the first few hundred steps."""
    parsed = parse_completion('work\n</think>\nThe answer is {"k": 7}')
    assert parsed.artifact == {"k": 7}
    assert "unfenced_answer" in parsed.violations
    assert not parsed.well_formed


def test_answer_only_inside_the_thinking_block_is_not_graded():
    parsed = parse_completion('I think ```json\n{"k": 7}\n```\n</think>\nno answer')
    assert parsed.artifact is None


def test_unparseable_json_is_not_an_exception():
    parsed = parse_completion('work\n</think>\n```json\n{"k": 07,}\n```')
    assert parsed.artifact is None
    assert "unparseable_json" in parsed.violations


def test_message_list_completions_are_accepted():
    parsed = parse_completion([{"role": "assistant", "content": 'w\n</think>\n```json\n{"k": 7}\n```'}])
    assert parsed.artifact == {"k": 7}


# --- reward arithmetic -------------------------------------------------------


def _reward_total(completion, task, weights=None) -> float:
    funcs = build_reward_functions(weights)
    kwargs = {
        "family": [task.family],
        "solution_json": [json.dumps(task.solution)],
        "params_json": [json.dumps(task.params)],
    }
    return sum(f([completion], **kwargs)[0] for f in funcs)


def _wrap(artifact: dict, think: str = "some reasoning") -> str:
    return f"{think}\n</think>\n```json\n{json.dumps(artifact)}\n```"


def test_a_correct_answer_beats_every_incorrect_one():
    task = generate("caesar", 0, 0)
    truth = task.solution["plaintext"]
    correct = _reward_total(_wrap({"plaintext": truth, "key": task.solution["key"]}), task)
    near_miss = _reward_total(_wrap({"plaintext": truth[:-2] + "QQ", "key": task.solution["key"]}), task)
    echo = _reward_total(_wrap({"plaintext": task.params["ciphertext"]}), task)
    junk = _reward_total("no thinking, no json at all", task)

    assert correct > near_miss > junk
    assert echo < junk or echo <= 0.0


def test_reward_weights_reject_a_config_where_a_near_miss_can_win():
    with pytest.raises(ValueError):
        RewardWeights(correctness=1.0, partial=0.8, format=0.5).check()


def test_thinking_budget_penalises_only_the_overrun():
    task = generate("caesar", 0, 0)
    artifact = {"plaintext": task.solution["plaintext"], "key": task.solution["key"]}
    weights = RewardWeights(think_budget_chars=100)
    short = _reward_total(_wrap(artifact, "x" * 50), task, weights)
    long = _reward_total(_wrap(artifact, "x" * 5000), task, weights)
    assert short > long
    assert short - long == pytest.approx(weights.budget_penalty)


def test_reward_functions_return_one_float_per_completion():
    task = generate("rsa-factor", 0, 0)
    kwargs = {
        "family": [task.family] * 3,
        "solution_json": [json.dumps(task.solution)] * 3,
        "params_json": [json.dumps(task.params)] * 3,
    }
    completions = [_wrap(task.solution), _wrap({"p": 2, "q": 3}), "garbage"]
    for func in build_reward_functions():
        scores = func(completions, **kwargs)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


# --- dataset -----------------------------------------------------------------


def test_train_and_eval_instances_cannot_overlap():
    train, evaluation = build_splits(200, 200)
    assert EVAL_SEED_BASE > 200
    assert not {r["task_id"] for r in train} & {r["task_id"] for r in evaluation}


def test_records_carry_everything_the_reward_needs():
    train, _ = build_splits(20, 4)
    for record in train:
        _, verdict = grade(
            _wrap(json.loads(record["solution_json"])),
            record["family"],
            json.loads(record["solution_json"]),
            json.loads(record["params_json"]),
        )
        assert verdict.accepted, record["family"]
