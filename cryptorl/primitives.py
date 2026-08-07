"""Cipher and number-theory primitives shared by the generators and verifiers.

Standard library only, on purpose. The generators and the reward verifiers are
the one part of an RL pipeline that must run everywhere -- in CI, on a laptop,
inside a trainer process that already owns the GPU -- and must produce the same
answer in all three. A dependency on gmpy2 or pycryptodome would buy speed at
the cost of that.

The secp256k1 arithmetic here is deliberately a second implementation of the
curve math already pinned in ``examples/ecdlp/checkers/secp256k1_dlog.py``.
``tests/test_cryptorl.py`` checks the two agree; if they ever disagree, this
file is the one that is wrong, because that one is hashed into a live
objective's id.
"""

from __future__ import annotations

import random

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# secp256k1
SECP256K1_P = 2**256 - 2**32 - 977
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP256K1_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
SECP256K1_G = (SECP256K1_GX, SECP256K1_GY)

Point = tuple[int, int] | None


def normalize_text(text: str) -> str:
    """Letters only, uppercased. The comparison basis for every classical task.

    Plaintext recovery is judged on letters because that is all a classical
    cipher transports; punishing a model for guessing where the spaces went
    would make the reward measure English formatting rather than cryptanalysis.
    """
    return "".join(ch for ch in text.upper() if ch in ALPHABET)


# --- classical ciphers -------------------------------------------------------


def caesar_encrypt(plaintext: str, shift: int) -> str:
    text = normalize_text(plaintext)
    return "".join(ALPHABET[(ALPHABET.index(c) + shift) % 26] for c in text)


def vigenere_encrypt(plaintext: str, key: str) -> str:
    text = normalize_text(plaintext)
    key = normalize_text(key)
    out = []
    for i, c in enumerate(text):
        k = ALPHABET.index(key[i % len(key)])
        out.append(ALPHABET[(ALPHABET.index(c) + k) % 26])
    return "".join(out)


def substitution_alphabet(keyword: str) -> str:
    """Keyword-derived cipher alphabet, so the key is a short recoverable string."""
    seen: list[str] = []
    for c in normalize_text(keyword):
        if c not in seen:
            seen.append(c)
    for c in ALPHABET:
        if c not in seen:
            seen.append(c)
    return "".join(seen)


def substitution_encrypt(plaintext: str, cipher_alphabet: str) -> str:
    text = normalize_text(plaintext)
    return "".join(cipher_alphabet[ALPHABET.index(c)] for c in text)


def columnar_encrypt(plaintext: str, key: str) -> str:
    """Columnar transposition, reading columns in alphabetical order of the key.

    The plaintext is trimmed to a whole number of rows rather than padded. A
    padded instance would make the ground-truth plaintext end in filler the
    solver has to guess is filler, which grades punctuation instinct, not
    transposition.
    """
    text = columnar_trim(plaintext, len(key))
    cols = len(key)
    rows = len(text) // cols
    order = sorted(range(cols), key=lambda i: (key[i], i))
    return "".join("".join(text[r * cols + c] for r in range(rows)) for c in order)


def columnar_trim(plaintext: str, cols: int) -> str:
    text = normalize_text(plaintext)
    return text[: (len(text) // cols) * cols]


# --- number theory -----------------------------------------------------------


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


def invmod(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("not invertible")
    return x % m


def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Deterministic for every n this pipeline generates (< 2^64) and a strong
    # probabilistic test above that; the RSA families never exceed 2^128.
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(rng: random.Random, bits: int) -> int:
    lo, hi = 1 << (bits - 1), (1 << bits) - 1
    while True:
        candidate = rng.randrange(lo, hi) | 1
        if is_probable_prime(candidate):
            return candidate


def integer_root(value: int, degree: int) -> int:
    """Floor of the degree-th root, exactly. No floats: a float cube root of a
    50-bit integer is off by one often enough to make a correct answer verify
    as wrong."""
    if value < 0:
        raise ValueError("negative radicand")
    if value < 2:
        return value
    guess = 1 << ((value.bit_length() + degree - 1) // degree + 1)
    while True:
        nxt = ((degree - 1) * guess + value // guess ** (degree - 1)) // degree
        if nxt >= guess:
            return guess
        guess = nxt


# --- secp256k1 ---------------------------------------------------------------


def ec_add(p: Point, q: Point) -> Point:
    if p is None:
        return q
    if q is None:
        return p
    (x1, y1), (x2, y2) = p, q
    if x1 == x2 and (y1 + y2) % SECP256K1_P == 0:
        return None
    if p == q:
        lam = (3 * x1 * x1) * pow(2 * y1, SECP256K1_P - 2, SECP256K1_P) % SECP256K1_P
    else:
        lam = (
            (y2 - y1)
            * pow((x2 - x1) % SECP256K1_P, SECP256K1_P - 2, SECP256K1_P)
            % SECP256K1_P
        )
    x3 = (lam * lam - x1 - x2) % SECP256K1_P
    return (x3, (lam * (x1 - x3) - y1) % SECP256K1_P)


def ec_mul(k: int, point: Point = SECP256K1_G) -> Point:
    result: Point = None
    addend = point
    while k:
        if k & 1:
            result = ec_add(result, addend)
        addend = ec_add(addend, addend)
        k >>= 1
    return result
