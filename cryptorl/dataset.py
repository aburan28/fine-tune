"""Turn generated tasks into rows a GRPO trainer can consume.

``solution`` and ``params`` are stored as JSON *strings*. They have different
keys in every family -- ``{"k"}`` here, ``{"n", "e1", "c1", ...}`` there -- and
an Arrow-backed dataset cannot hold a column whose schema changes per row. The
reward functions decode them; nothing else looks at them.

Train and eval split by seed range rather than by shuffling, because the tasks
are a pure function of the seed. Non-overlapping ranges make leakage
impossible instead of merely unlikely.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .tasks import CLASSICAL_FAMILIES, FAMILIES, MAX_DIFFICULTY, generate

EVAL_SEED_BASE = 10_000_000
"""Eval seeds start here. Training never draws above it."""


def build_records(
    size: int,
    *,
    families: tuple[str, ...] = FAMILIES,
    difficulties: tuple[int, ...] = tuple(range(MAX_DIFFICULTY + 1)),
    seed_base: int = 0,
    shuffle_seed: int = 0,
) -> list[dict]:
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")

    records = []
    for i in range(size):
        family = families[i % len(families)]
        difficulty = difficulties[(i // len(families)) % len(difficulties)]
        task = generate(family, difficulty, seed_base + i)
        records.append(
            {
                "prompt": task.messages(),
                "task_id": task.task_id,
                "family": task.family,
                "difficulty": task.difficulty,
                "solution_json": json.dumps(task.solution),
                "params_json": json.dumps(task.params),
            }
        )
    random.Random(shuffle_seed).shuffle(records)
    return records


def build_splits(train_size: int, eval_size: int, **kwargs) -> tuple[list[dict], list[dict]]:
    train = build_records(train_size, seed_base=0, **kwargs)
    evaluation = build_records(eval_size, seed_base=EVAL_SEED_BASE, **kwargs)
    return train, evaluation


def to_hf_dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(records)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="directory for train.jsonl / eval.jsonl")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILIES),
        help=f"subset of {list(FAMILIES)}; 'classical' expands to {list(CLASSICAL_FAMILIES)}",
    )
    parser.add_argument("--max-difficulty", type=int, default=MAX_DIFFICULTY)
    args = parser.parse_args()

    families = tuple(
        f for name in args.families for f in (CLASSICAL_FAMILIES if name == "classical" else (name,))
    )
    train, evaluation = build_splits(
        args.train_size,
        args.eval_size,
        families=families,
        difficulties=tuple(range(args.max_difficulty + 1)),
    )
    write_jsonl(train, Path(args.out) / "train.jsonl")
    write_jsonl(evaluation, Path(args.out) / "eval.jsonl")
    print(f"wrote {len(train)} train and {len(evaluation)} eval rows to {args.out}/")


if __name__ == "__main__":
    main()
