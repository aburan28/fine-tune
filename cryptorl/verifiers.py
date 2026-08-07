"""Deterministic verifiers. These are the reward signal; nothing else is.

Each returns a ``Verdict``. The convention -- ``(accepted, reason)`` from a
pinned, side-effect-free checker -- is lifted from this repository's own
verifiers so a policy trained here produces artifacts of exactly the shape
``score_candidate`` already grades. See ``docs/verification.md``.

Three rules carried over from the network's checkers, because they are what
keeps a reward function from teaching the wrong thing:

1. **A malformed artifact is a rejection, not an exception.** Raising would
   crash the trainer on a rollout whose only crime is being bad output; the
   whole point of RL is that most early rollouts are bad output.
2. **Unverifiable is not rejected.** Not reachable here -- every verifier is
   pure arithmetic -- but the invariant is why none of these consults a clock,
   a file, or a network.
3. **The verifier decides, not the model.** Nothing reads the model's own claim
   about whether it succeeded. A ``"confidence"`` or ``"solved"`` field in an
   artifact is ignored on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import primitives as prim

# Certificate families verify from the instance alone. Recovery families need
# the plaintext that was actually encrypted, because "correct" is defined as
# matching it.
CERTIFICATE_FAMILIES = frozenset(
    {
        "dlp-modp",
        "dlp-secp256k1",
        "rsa-factor",
        "rsa-cube",
        "rsa-common-modulus",
        "rsa-wiener",
    }
)


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str
    partial: float = 0.0
    """Structural credit in [0, 1]. Advisory only: it never turns a rejection
    into an acceptance, and the reward weights keep the best possible partial
    score strictly below an exact solve."""
    invalid: bool = False
    """The artifact broke a constraint the prompt stated -- a non-integer where
    an integer was required, a scalar outside the given bound, a factor of one.
    Distinguished from an ordinary wrong answer because the two deserve
    different gradients: being wrong is the job, ignoring the problem statement
    is not."""


def _int_field(artifact: dict, name: str) -> int | None:
    value = artifact.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _char_accuracy(guess: str, truth: str) -> float:
    if not truth:
        return 0.0
    matches = sum(1 for a, b in zip(guess, truth) if a == b)
    return matches / len(truth)


# --- recovery families -------------------------------------------------------


def _verify_classical(artifact: dict, solution: dict, params: dict) -> Verdict:
    raw = artifact.get("plaintext")
    if not isinstance(raw, str):
        return Verdict(False, "artifact.plaintext must be a string", invalid=True)

    guess = prim.normalize_text(raw)
    truth = solution["plaintext"]
    ciphertext = params["ciphertext"]

    if guess == prim.normalize_text(ciphertext) and guess != truth:
        # Echoing the ciphertext scores well on any position-wise metric for a
        # transposition (same multiset of letters) and is the single most
        # available degenerate strategy. It earns nothing.
        return Verdict(False, "plaintext is the ciphertext unchanged", invalid=True)

    key_guess = artifact.get("key")
    key_ok = isinstance(key_guess, str) and key_guess.strip().upper() == solution["key"].upper()

    if guess == truth:
        return Verdict(True, "plaintext recovered" + (" with key" if key_ok else ""), partial=1.0)

    partial = 0.7 * _char_accuracy(guess, truth) + (0.3 if key_ok else 0.0)
    return Verdict(False, "plaintext does not match", partial=partial)


# --- certificate families ----------------------------------------------------


def _verify_dlp_modp(artifact: dict, params: dict) -> Verdict:
    k = _int_field(artifact, "k")
    if k is None:
        return Verdict(False, "artifact.k must be an integer", invalid=True)
    if not 0 <= k < params["order"]:
        return Verdict(False, f"k must lie in [0, {params['order']})", invalid=True)
    if pow(params["g"], k, params["p"]) != params["h"]:
        return Verdict(False, "g^k != h (mod p)", partial=0.2)
    return Verdict(True, "g^k = h (mod p)", partial=1.0)


def _verify_dlp_secp256k1(artifact: dict, params: dict) -> Verdict:
    k = _int_field(artifact, "k")
    if k is None:
        return Verdict(False, "artifact.k must be an integer", invalid=True)
    if not params["low"] <= k < params["high"]:
        return Verdict(False, f"k outside the stated bound; got a {k.bit_length()}-bit value", invalid=True)
    point = prim.ec_mul(k)
    if point is None:
        return Verdict(False, "k*G is the point at infinity", invalid=True)
    if point != (params["x"], params["y"]):
        return Verdict(False, "k*G does not equal P", partial=0.2)
    return Verdict(True, "k*G = P", partial=1.0)


def _verify_rsa_factor(artifact: dict, params: dict) -> Verdict:
    n = params["n"]
    p, q = _int_field(artifact, "p"), _int_field(artifact, "q")
    if p is None or q is None:
        return Verdict(False, "artifact.p and artifact.q must be integers", invalid=True)
    if p <= 1 or q <= 1:
        return Verdict(False, "factors must exceed 1", invalid=True)
    if p * q != n:
        # A single correct factor is most of the work and is checkable without
        # knowing the answer, so it is the one place partial credit here is real.
        divides = any(1 < f < n and n % f == 0 for f in (p, q))
        return Verdict(False, "p*q != n", partial=0.5 if divides else 0.0)
    return Verdict(True, "p*q = n", partial=1.0)


def _verify_rsa_cube(artifact: dict, params: dict) -> Verdict:
    m = _int_field(artifact, "m")
    if m is None:
        return Verdict(False, "artifact.m must be an integer", invalid=True)
    if not 1 < m < params["n"]:
        return Verdict(False, "m must lie in (1, n)", invalid=True)
    if m**3 != params["c"]:
        return Verdict(False, "m^3 != c", partial=0.2)
    return Verdict(True, "m^3 = c", partial=1.0)


def _verify_rsa_common_modulus(artifact: dict, params: dict) -> Verdict:
    m = _int_field(artifact, "m")
    if m is None:
        return Verdict(False, "artifact.m must be an integer", invalid=True)
    n = params["n"]
    if not 1 < m < n:
        return Verdict(False, "m must lie in (1, n)", invalid=True)
    if pow(m, params["e1"], n) != params["c1"] or pow(m, params["e2"], n) != params["c2"]:
        return Verdict(False, "m does not reproduce both ciphertexts", partial=0.2)
    return Verdict(True, "m reproduces both ciphertexts", partial=1.0)


# Fixed probes, so the verdict does not depend on a random draw. Any d that
# decrypts these is a usable private exponent even if it is not the d that was
# generated -- d is only unique modulo lcm(p-1, q-1), and rejecting a working
# key because it is not the one we happened to pick would be a lie.
#
# The d < n bound below is the canonical range for a private exponent, and it
# also caps the cost of the check: without it a rollout could submit a
# million-bit integer and make the reward function, which runs once per rollout
# per training step, the slowest thing in the loop.
_WIENER_PROBES = (2, 3, 5, 7)


def _verify_rsa_wiener(artifact: dict, params: dict) -> Verdict:
    d = _int_field(artifact, "d")
    if d is None:
        return Verdict(False, "artifact.d must be an integer", invalid=True)
    n, e = params["n"], params["e"]
    if not 1 < d < n:
        return Verdict(False, "d must lie in (1, n)", invalid=True)
    for m in _WIENER_PROBES:
        if pow(pow(m, e, n), d, n) != m % n:
            return Verdict(False, "d does not invert e", partial=0.2)
    return Verdict(True, "d inverts e on every probe", partial=1.0)


_CERTIFICATE_VERIFIERS = {
    "dlp-modp": _verify_dlp_modp,
    "dlp-secp256k1": _verify_dlp_secp256k1,
    "rsa-factor": _verify_rsa_factor,
    "rsa-cube": _verify_rsa_cube,
    "rsa-common-modulus": _verify_rsa_common_modulus,
    "rsa-wiener": _verify_rsa_wiener,
}


def verify(family: str, artifact: object, solution: dict, params: dict) -> Verdict:
    if not isinstance(artifact, dict):
        return Verdict(False, "artifact must be a JSON object", invalid=True)
    if family in _CERTIFICATE_VERIFIERS:
        return _CERTIFICATE_VERIFIERS[family](artifact, params)
    if family in ("caesar", "vigenere", "substitution", "transposition"):
        return _verify_classical(artifact, solution, params)
    raise ValueError(f"unknown family {family!r}")


def verify_task(task, artifact: object) -> Verdict:
    return verify(task.family, artifact, task.solution, task.params)
