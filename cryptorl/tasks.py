"""Seeded generators for the cryptanalysis curriculum.

Every task is a pure function of ``(family, difficulty, seed)``. That is what
makes the training set reproducible and, more importantly, what makes the
held-out evaluation set *actually* held out: an eval seed range that never
overlaps a training seed range cannot leak instances, no shuffling discipline
required.

Two shapes of task live here, and the difference matters for what the reward
can honestly measure:

- **Certificate families** (``dlp-*``, ``rsa-*``): the verifier re-derives the
  answer from the instance alone and never consults the stored solution. This
  is the shape the network in this repository is built around -- see
  ``examples/ecdlp/checkers/secp256k1_dlog.py``.
- **Recovery families** (the classical ciphers): "correct" is defined as
  matching the plaintext that was actually encrypted, so the verifier must hold
  it. Partial credit is only meaningful here.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from . import primitives as prim

# Original filler prose. The corpus only has to be English-shaped: it supplies
# the letter statistics that make frequency analysis work at all.
CORPUS = [
    "The harbour lights came on one by one as the tide turned against the wall.",
    "Every measurement we trusted last winter turned out to depend on the same broken clock.",
    "She kept the ledger in a drawer that nobody else in the building had a key for.",
    "A quiet river will carve a canyon given enough centuries and no interruption.",
    "The map was accurate everywhere except the one valley we actually had to cross.",
    "Nothing in the report was false, and nothing in it was the whole truth either.",
    "He learned to read the weather from the behaviour of birds along the cliff.",
    "The archive burned in the spring, so every later account depends on one copy.",
    "We agreed the experiment had failed before we agreed on what it had been testing.",
    "A locked door tells you less about the room than the wear on the floor beside it.",
    "The engine ran perfectly for eleven hours and then stopped without any warning.",
    "Most of the village had moved downstream by the time the second bridge opened.",
    "Her notebooks contain three separate proofs, and only the shortest is correct.",
    "The signal repeated every ninety seconds for a week and then never came again.",
    "Old charts marked the shoal in the wrong place, which is why the wreck is there.",
    "They counted the rings and found the drought had lasted longer than the records said.",
    "A single mistranslated word kept the two delegations arguing for another month.",
    "The library catalogue was reorganised twice and lost more books each time.",
    "Snow covered the tracks before anyone thought to photograph them properly.",
    "We can reconstruct the route from fuel receipts but not from anybody's memory.",
    "The instrument was sensitive enough to detect the train passing half a mile away.",
    "Each generation rebuilt the wall a little further inland than the last one had.",
    "No one recorded the name of the engineer who noticed the crack in the casting.",
    "The letters stop abruptly in autumn and resume in a different hand entirely.",
    "A good forgery fails on the paper long before it fails on the handwriting.",
    "The census undercounted the district by roughly a third for twenty years running.",
    "Sunlight through the broken roof had bleached half the mural into plain stone.",
    "He wrote the key on the inside of the cabinet door and then forgot the cabinet.",
    "The tunnel was dug from both ends and the halves missed each other by a metre.",
    "Everything they buried in the field was found again by a farmer the next decade.",
]

KEYWORDS = [
    "LANTERN", "HARBOUR", "CIPHER", "MERIDIAN", "GRANITE", "TEMPEST", "ORCHARD",
    "BASALT", "QUARRY", "SEXTANT", "FURNACE", "MIDNIGHT", "COMPASS", "THICKET",
    "VELLUM", "ANVIL", "PLUMAGE", "SANDBAR", "TRELLIS", "WAXWING",
]

FAMILIES = (
    "caesar",
    "vigenere",
    "substitution",
    "transposition",
    "dlp-modp",
    "dlp-secp256k1",
    "rsa-factor",
    "rsa-cube",
    "rsa-common-modulus",
    "rsa-wiener",
)

CLASSICAL_FAMILIES = ("caesar", "vigenere", "substitution", "transposition")

MAX_DIFFICULTY = 3

SYSTEM_PROMPT = (
    "You are a cryptanalyst. Work the problem inside <think> and </think>: state "
    "what you know, try an approach, and check it against the data before you "
    "commit. Then give your final answer as a single JSON object inside one "
    "```json fenced block after </think>. That JSON object is the only thing "
    "that is graded. Write exactly one: a response containing several candidate "
    "answers scores zero, however good one of them is."
)


@dataclass(frozen=True)
class Task:
    family: str
    difficulty: int
    seed: int
    prompt: str
    solution: dict
    params: dict = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        raw = f"{self.family}:{self.difficulty}:{self.seed}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def messages(self) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.prompt},
        ]


def _plaintext(rng: random.Random, length: int) -> str:
    pool = CORPUS[:]
    rng.shuffle(pool)
    parts, total, i = [], 0, 0
    while total < length:
        chunk = prim.normalize_text(pool[i % len(pool)])
        parts.append(chunk)
        total += len(chunk)
        i += 1
    return "".join(parts)[:length]


def _answer_shape(shape: str) -> str:
    return f"Answer with exactly this JSON shape: {shape}"


# --- classical ---------------------------------------------------------------

_CLASSICAL_LENGTH = {0: 60, 1: 110, 2: 180, 3: 260}


def _gen_caesar(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    plaintext = _plaintext(rng, _CLASSICAL_LENGTH[difficulty])
    shift = rng.randrange(1, 26)
    ciphertext = prim.caesar_encrypt(plaintext, shift)
    prompt = (
        "This ciphertext is a Caesar shift of English text over the 26-letter "
        "alphabet, spaces and punctuation removed.\n\n"
        f"CIPHERTEXT:\n{ciphertext}\n\n"
        "Recover the plaintext and the shift.\n"
        + _answer_shape('{"plaintext": "<letters only, uppercase>", "key": "<shift as a number 1-25>"}')
    )
    return prompt, {"plaintext": plaintext, "key": str(shift)}, {"ciphertext": ciphertext}


def _gen_vigenere(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    plaintext = _plaintext(rng, _CLASSICAL_LENGTH[difficulty] * 2)
    key_len = {0: 3, 1: 4, 2: 6, 3: 8}[difficulty]
    key = "".join(rng.choice(prim.ALPHABET) for _ in range(key_len))
    ciphertext = prim.vigenere_encrypt(plaintext, key)
    prompt = (
        "This ciphertext is a Vigenere encryption of English text over the "
        "26-letter alphabet, spaces and punctuation removed. The key is "
        f"{key_len} letters long.\n\n"
        f"CIPHERTEXT:\n{ciphertext}\n\n"
        "Recover the plaintext and the key.\n"
        + _answer_shape('{"plaintext": "<letters only, uppercase>", "key": "<the key, uppercase>"}')
    )
    return prompt, {"plaintext": plaintext, "key": key}, {"ciphertext": ciphertext, "key_length": key_len}


def _gen_substitution(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    plaintext = _plaintext(rng, _CLASSICAL_LENGTH[difficulty] * 3)
    keyword = rng.choice(KEYWORDS)
    cipher_alphabet = prim.substitution_alphabet(keyword)
    ciphertext = prim.substitution_encrypt(plaintext, cipher_alphabet)
    prompt = (
        "This ciphertext is a monoalphabetic substitution of English text, "
        "spaces and punctuation removed. The cipher alphabet was built from an "
        "English keyword: the distinct letters of the keyword first, then the "
        "unused letters in alphabetical order.\n\n"
        f"CIPHERTEXT:\n{ciphertext}\n\n"
        "Recover the plaintext and the cipher alphabet.\n"
        + _answer_shape(
            '{"plaintext": "<letters only, uppercase>", '
            '"key": "<26-letter cipher alphabet: the letter that A maps to, then B, ...>"}'
        )
    )
    return (
        prompt,
        {"plaintext": plaintext, "key": cipher_alphabet},
        {"ciphertext": ciphertext, "keyword": keyword},
    )


def _gen_transposition(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    cols = {0: 4, 1: 5, 2: 6, 3: 7}[difficulty]
    key = "".join(rng.sample(prim.ALPHABET, cols))
    plaintext = prim.columnar_trim(_plaintext(rng, _CLASSICAL_LENGTH[difficulty] * 2), cols)
    ciphertext = prim.columnar_encrypt(plaintext, key)
    prompt = (
        "This ciphertext is a columnar transposition of English text, spaces and "
        f"punctuation removed. The key has {cols} distinct letters; the plaintext "
        "was written across the rows and the columns were then read out in "
        "alphabetical order of their key letters. The plaintext length is an exact "
        "multiple of the number of columns, so there is no padding.\n\n"
        f"CIPHERTEXT:\n{ciphertext}\n\n"
        "Recover the plaintext and the key.\n"
        + _answer_shape('{"plaintext": "<letters only, uppercase>", "key": "<the key word, uppercase>"}')
    )
    return prompt, {"plaintext": plaintext, "key": key}, {"ciphertext": ciphertext, "columns": cols}


# --- discrete logs -----------------------------------------------------------

# Safe primes p = 2q+1, so the order of the subgroup a square generates is
# exactly q and can be stated in the prompt rather than guessed at.
_SAFE_PRIMES = {0: 4127, 1: 131267, 2: 16777907, 3: 2147483783}
_DLP_BITS = {0: 12, 1: 16, 2: 20, 3: 24}


def _gen_dlp_modp(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    p = _SAFE_PRIMES[difficulty]
    q = (p - 1) // 2
    # A generator of the order-q subgroup: any non-identity square.
    while True:
        g = pow(rng.randrange(2, p - 1), 2, p)
        if g != 1:
            break
    k = rng.randrange(2, q)
    h = pow(g, k, p)
    prompt = (
        "Solve a discrete logarithm in the multiplicative group modulo a prime.\n\n"
        f"p = {p}\ng = {g}\nh = {h}\n\n"
        f"g has prime order q = {q} modulo p. Find the integer k with 0 <= k < q "
        "such that g^k = h (mod p).\n"
        + _answer_shape('{"k": <integer>}')
    )
    return prompt, {"k": k}, {"p": p, "g": g, "h": h, "order": q}


def _gen_dlp_secp256k1(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    bits = _DLP_BITS[difficulty]
    low, high = 1 << bits, 1 << (bits + 1)
    k = rng.randrange(low, high)
    point = prim.ec_mul(k)
    assert point is not None
    prompt = (
        "Recover a bounded discrete logarithm on the secp256k1 curve.\n\n"
        f"P.x = {point[0]}\nP.y = {point[1]}\n\n"
        f"P = k*G for the standard secp256k1 generator G, and k lies in "
        f"[2^{bits}, 2^{bits + 1}). Find k.\n"
        + _answer_shape('{"k": <integer>}')
    )
    return prompt, {"k": k}, {"x": point[0], "y": point[1], "low": low, "high": high}


# --- RSA ---------------------------------------------------------------------

_RSA_PRIME_BITS = {0: 12, 1: 16, 2: 20, 3: 24}


def _gen_rsa_factor(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    bits = _RSA_PRIME_BITS[difficulty]
    p = prim.random_prime(rng, bits)
    q = prim.random_prime(rng, bits)
    while q == p:
        q = prim.random_prime(rng, bits)
    n = p * q
    prompt = (
        "Factor an RSA modulus.\n\n"
        f"n = {n}\ne = 65537\n\n"
        f"n is the product of two distinct {bits}-bit primes. Find them.\n"
        + _answer_shape('{"p": <integer>, "q": <integer>}')
    )
    return prompt, {"p": min(p, q), "q": max(p, q)}, {"n": n, "e": 65537}


def _gen_rsa_cube(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    bits = _RSA_PRIME_BITS[difficulty] + 8
    p = prim.random_prime(rng, bits * 2)
    q = prim.random_prime(rng, bits * 2)
    n = p * q
    # m^3 < n, so the modulus never wraps and the ciphertext is a perfect cube.
    m = rng.randrange(1 << (bits - 1), 1 << bits)
    c = m**3
    assert c < n
    prompt = (
        "An RSA message was encrypted with a small public exponent and no "
        "padding.\n\n"
        f"n = {n}\ne = 3\nc = {c}\n\n"
        "The message is short enough that m^3 never exceeded n. Recover m.\n"
        + _answer_shape('{"m": <integer>}')
    )
    return prompt, {"m": m}, {"n": n, "e": 3, "c": c}


def _gen_rsa_common_modulus(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    bits = _RSA_PRIME_BITS[difficulty] + 12
    p = prim.random_prime(rng, bits)
    q = prim.random_prime(rng, bits)
    n = p * q
    phi = (p - 1) * (q - 1)
    while True:
        e1 = rng.randrange(3, 1 << 16) | 1
        e2 = rng.randrange(3, 1 << 16) | 1
        if e1 != e2 and prim.egcd(e1, e2)[0] == 1:
            if prim.egcd(e1, phi)[0] == 1 and prim.egcd(e2, phi)[0] == 1:
                break
    m = rng.randrange(2, n - 1)
    c1, c2 = pow(m, e1, n), pow(m, e2, n)
    prompt = (
        "The same message was sent twice under the same RSA modulus with two "
        "different public exponents.\n\n"
        f"n = {n}\ne1 = {e1}\nc1 = {c1}\ne2 = {e2}\nc2 = {c2}\n\n"
        "Recover m.\n" + _answer_shape('{"m": <integer>}')
    )
    return prompt, {"m": m}, {"n": n, "e1": e1, "c1": c1, "e2": e2, "c2": c2}


def _gen_rsa_wiener(rng: random.Random, difficulty: int) -> tuple[str, dict, dict]:
    bits = {0: 32, 1: 48, 2: 64, 3: 80}[difficulty]
    while True:
        p = prim.random_prime(rng, bits)
        q = prim.random_prime(rng, bits)
        if p == q:
            continue
        # Wiener's bound needs q < p < 2q as well as a small d.
        if not (q < p < 2 * q):
            continue
        n = p * q
        phi = (p - 1) * (q - 1)
        bound = prim.integer_root(n, 4) // 3
        if bound < 4:
            continue
        d = rng.randrange(bound // 2, bound) | 1
        if prim.egcd(d, phi)[0] != 1:
            continue
        e = prim.invmod(d, phi)
        # A small e would be recoverable without the continued-fraction attack.
        if e.bit_length() < n.bit_length() - 8:
            continue
        break
    prompt = (
        "An RSA key was generated with a private exponent chosen to make "
        "decryption fast.\n\n"
        f"n = {n}\ne = {e}\n\n"
        "Recover the private exponent d.\n" + _answer_shape('{"d": <integer>}')
    )
    return prompt, {"d": d}, {"n": n, "e": e}


_GENERATORS = {
    "caesar": _gen_caesar,
    "vigenere": _gen_vigenere,
    "substitution": _gen_substitution,
    "transposition": _gen_transposition,
    "dlp-modp": _gen_dlp_modp,
    "dlp-secp256k1": _gen_dlp_secp256k1,
    "rsa-factor": _gen_rsa_factor,
    "rsa-cube": _gen_rsa_cube,
    "rsa-common-modulus": _gen_rsa_common_modulus,
    "rsa-wiener": _gen_rsa_wiener,
}


def generate(family: str, difficulty: int, seed: int) -> Task:
    if family not in _GENERATORS:
        raise ValueError(f"unknown family {family!r}")
    if not 0 <= difficulty <= MAX_DIFFICULTY:
        raise ValueError(f"difficulty must be 0..{MAX_DIFFICULTY}")
    # Seeding on the triple, not on `seed` alone, keeps the families independent:
    # otherwise seed 7 draws the same random stream in every family and the
    # curriculum correlates across tasks it has no reason to correlate across.
    rng = random.Random(f"cryptorl/{family}/{difficulty}/{seed}")
    prompt, solution, params = _GENERATORS[family](rng, difficulty)
    return Task(
        family=family,
        difficulty=difficulty,
        seed=seed,
        prompt=prompt,
        solution=solution,
        params=params,
    )
