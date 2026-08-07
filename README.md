# GRPO fine-tuning for cryptanalysis

Reinforcement learning on a Qwen reasoning model, with **pinned deterministic
verifiers as the entire reward signal**. No reward model, no preference data,
no human labels — a rollout either recovers the plaintext, the discrete log or
the factorisation, or it does not, and a checker that runs in microseconds says
which.

The bet is that verification is cheap by construction, so the verifier can be
an inner-loop fitness function rather than an audit step. That idea is borrowed
from [proofwork](https://github.com/aburan28/distributed-researcher), where
pinned checkers grade submitted results and
[`docs/agents.md`](https://github.com/aburan28/distributed-researcher/blob/main/docs/agents.md)
makes the same argument for agents; this repository applies it to gradient
descent. The artifact shapes match, so a policy trained here can be pointed at
a live objective without a translation layer.

## Layout

| path | needs a GPU | what it is |
|---|---|---|
| [`cryptorl/tasks.py`](cryptorl/tasks.py) | no | seeded instance generators for ten cipher families |
| [`cryptorl/verifiers.py`](cryptorl/verifiers.py) | no | the reward ground truth |
| [`cryptorl/rewards.py`](cryptorl/rewards.py) | no | completion parsing and TRL reward callables |
| [`cryptorl/dataset.py`](cryptorl/dataset.py) | no | task → training row, and the train/eval split |
| [`train_grpo.py`](train_grpo.py) | yes | TRL `GRPOTrainer` + LoRA |
| [`evaluate.py`](evaluate.py) | yes | pass@k on held-out instances, by family and difficulty |
| [`aws/run-on-ec2.sh`](aws/README.md) | — | launch the run on an EC2 spot GPU, end to end |

The split down that column is deliberate. The reward pipeline is the part that
decides what the model learns, so it is the part that has to be testable
without a training rig:

```bash
python3 -m pytest tests/ -q
```

144 tests, no dependencies beyond `pytest`, well under a second. Two of them
compare this repository's secp256k1 arithmetic against the copy pinned into a
live proofwork objective and skip unless that checkout is nearby; point
`PROOFWORK_ROOT` at it to run them.

## Quick start

No GPU handy? [`aws/`](aws/README.md) launches the whole thing on an EC2 spot
instance — roughly $0.63/hr for a 24GB A10G, and built to resume after an
interruption rather than to avoid one:

```bash
./aws/run-on-ec2.sh price     # what it would cost, launches nothing
./aws/run-on-ec2.sh up        # plan, confirm, then train end to end
```

On your own machine:

```bash
pip install -r requirements.txt
```

Prove the loop closes before spending a night on it — twenty steps on the
smallest Qwen3, about ten minutes:

```bash
python train_grpo.py --config configs/smoke.yaml
```

Then the real run:

```bash
python train_grpo.py --config configs/qwen3-4b-24gb.yaml
```

Generation, not the backward pass, is the wall clock here: 2000 steps is
~64k completions. Install `vllm` and set `use_vllm: true` if the run needs to
finish overnight rather than over a weekend.

Then measure it against the untrained model, which is the only comparison that
means anything:

```bash
python evaluate.py --model Qwen/Qwen3-4B --samples 4 --out baseline.json
python evaluate.py --model Qwen/Qwen3-4B --adapter runs/qwen3-4b-crypto/final --samples 4 --out tuned.json
```

## The task families

Ten families, four difficulty levels each, all generated from
`(family, difficulty, seed)` — so an evaluation seed range that never overlaps
a training seed range cannot leak instances, no shuffling discipline required.

**Recovery families** — success is defined as matching the plaintext that was
actually encrypted, so the verifier holds it:

- `caesar`, `vigenere` (key length 3–8), `substitution` (keyword alphabet),
  `transposition` (columnar, 4–7 columns)

**Certificate families** — the verifier re-derives the answer from the instance
alone and never consults the stored solution, exactly as
[`examples/ecdlp/checkers/secp256k1_dlog.py`](https://github.com/aburan28/distributed-researcher/blob/main/examples/ecdlp/checkers/secp256k1_dlog.py)
does:

- `dlp-modp` — discrete log in a prime-order subgroup, p from 4127 to 2³¹
- `dlp-secp256k1` — bounded ECDLP, k in [2¹², 2²⁵) by difficulty
- `rsa-factor` — semiprime factoring, 12- to 24-bit primes
- `rsa-cube` — unpadded e=3 where m³ < n
- `rsa-common-modulus` — same message, same modulus, coprime exponents
- `rsa-wiener` — small private exponent

`test_ec_mul_matches_the_pinned_checker` asserts this directory's secp256k1
arithmetic agrees with the copy hashed into the live objective's id. If they
ever disagree, this one is wrong.

## The reward

Five functions, summed by TRL. The weights are the actual design decision:

| function | weight | signal |
|---|---|---|
| `correctness_reward` | 2.0 | the verifier accepted |
| `partial_credit_reward` | 0.4 | structural progress (see below) |
| `format_reward` | 0.3 | one `<think>` block, one fenced JSON answer |
| `invalid_step_penalty` | −0.6 | broke a constraint the prompt stated |
| `thinking_budget_penalty` | −0.2 | reasoning past the budget |

`RewardWeights.check()` refuses a configuration in which a perfectly formatted
near-miss can out-earn a correct answer. That ordering is the one property the
whole scheme rests on, and it is easy to break by nudging a weight, so it is
asserted rather than documented.

### What partial credit can and cannot be

Per-character plaintext accuracy is a real gradient: a Vigenère key that is
right in three of six positions decrypts a third of the message, and rewarding
that is what lets the policy climb. A discrete log has no such structure —
being one off is exactly as wrong as being 2³⁹ off — so those families get a
flat 0.2 for a well-formed in-range answer and nothing else. Inventing a
smooth-looking signal there would reward guessing.

RSA factoring is the middle case: one correct factor is most of the work *and*
is checkable without knowing the answer, so it earns 0.5.

### Reward hacking this is built to resist

- **Echoing the ciphertext.** Scores well on any position-wise metric for a
  transposition, since the letters are the same ones. Detected and zeroed.
- **Shotgunning.** Emitting ten candidate answers and letting the grader find
  the good one is parser exploitation, not cryptanalysis. More than one JSON
  object after `</think>` scores zero even if one of them is correct.
- **Self-assessment.** A `"confidence"` or `"solved"` field in an artifact is
  ignored. Nothing in the reward path reads the model's own claim about whether
  it succeeded.
- **Answering inside the thinking block.** Not graded.

## Thinking budget

Qwen3's chat template opens `<think>` as part of the *prompt* when
`enable_thinking=True`, so a completion legitimately carries only the closing
tag. `parse_completion` accepts that; requiring the opening tag would zero the
format reward on every rollout and teach the model to emit a second, nested
block. `train_grpo.py` pins `enable_thinking=True` explicitly rather than
relying on the template default, so a future template change cannot silently
turn reasoning off — which would present as a reward collapse with no code
change to blame.

`max_completion_length: 2048` is the hard budget; `think_budget_chars: 6000`
applies soft pressure below it. Truncating mid-derivation produces an
unparseable answer and a wrongly-zero reward, so raise the hard limit before
tightening the soft one.

## Tuning notes

- **`num_generations`** is GRPO's group size. The advantage *is* the spread of
  reward within a group, so 4 is the floor and 8 is where it stops being noisy.
  On sparse tasks a group where every rollout fails contributes zero gradient —
  which is why the shipped config starts at `max_difficulty: 1`.
- **`beta: 0.02`** is the KL to the frozen reference. Cryptanalysis rewards are
  sparse and spiky; without it the policy finds a degenerate high-reward
  dialect and stops writing English.
- **Do not lower `temperature`.** Exploration is where the reward comes from.
- **Raise difficulty only after the solve rate moves.** A family that never
  earns reward contributes variance and nothing else. `dlp-secp256k1` at
  difficulty 2–3 and `rsa-wiener` at 3 are not reachable in-context for a 4B
  model; they are in the curriculum for larger bases and for measuring the
  ceiling, not for the first run.

## What this does not claim

The verifiers check *answers*, not *methods*. A model that recovers `k` by an
approach it cannot articulate scores identically to one that runs
baby-step giant-step correctly — which is the right call for a reward signal
(nothing about how a submitter found a discrete log is verifiable, and nothing
about it needs to be) but means the `<think>` traces are not evidence of
anything. Do not mine them for a distilled SFT set without checking them.

Nor is any of this a claim about real cryptography. Every instance is sized to
be solvable: 24-bit RSA primes, discrete logs bounded to 25 bits, classical
ciphers. Nothing here bears on the security of secp256k1 or of RSA at real
parameters.

## Using a trained model on the proofwork network

The artifact shapes are the same ones
[proofwork](https://github.com/aburan28/distributed-researcher)'s verifiers
grade, so a policy trained here can be pointed at a live objective through its
`proofwork` MCP tools without a translation layer — `score_candidate` runs the
pinned verifier for free and is the same reward signal this pipeline trains
against.
See [`docs/agents.md`](https://github.com/aburan28/distributed-researcher/blob/main/docs/agents.md) for the loop, and note that
`score_candidate` records nothing: it is the inner loop, and `submit_claim` is
the two-call commit–reveal that follows it.
