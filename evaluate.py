"""Score a checkpoint on the held-out split, broken down by family and difficulty.

    python evaluate.py --model Qwen/Qwen3-4B --adapter runs/.../final --samples 4

Run it once with ``--adapter`` omitted first. A single aggregate number cannot
tell you whether GRPO taught cryptanalysis or taught JSON formatting, so the
baseline is not optional -- and the per-family table is where a policy that got
better at Caesar while getting worse at RSA becomes visible.

pass@k here is the plain empirical estimate over ``--samples`` independent
rollouts at the training temperature. It is measured, not extrapolated from
pass@1, because the unbiased-estimator formula assumes independence the model
does not actually give you at temperature 1.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from cryptorl.dataset import build_splits
from cryptorl.rewards import grade
from cryptorl.tasks import CLASSICAL_FAMILIES, FAMILIES, MAX_DIFFICULTY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="base model id or path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter; omit for the baseline")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4, help="rollouts per instance (the k)")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--max-difficulty", type=int, default=MAX_DIFFICULTY)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    return parser.parse_args()


def load_model(model_id: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], args) -> list[str]:
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Slice off the prompt so the completion the grader sees is what the policy
    # actually produced, not the instructions it was given.
    return tokenizer.batch_decode(output[:, encoded["input_ids"].shape[1] :], skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    families = tuple(
        f for name in args.families for f in (CLASSICAL_FAMILIES if name == "classical" else (name,))
    )
    _, records = build_splits(
        1,
        args.size,
        families=families,
        difficulties=tuple(range(args.max_difficulty + 1)),
    )

    model, tokenizer = load_model(args.model, args.adapter)
    prompts = [
        tokenizer.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for r in records
    ]

    solved_any = defaultdict(int)
    solved_first = defaultdict(int)
    total = defaultdict(int)
    invalid = defaultdict(int)
    detail = []

    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        chunk_prompts = prompts[start : start + args.batch_size]
        per_record: list[list[bool]] = [[] for _ in chunk]
        per_record_invalid = [0] * len(chunk)
        for _ in range(args.samples):
            completions = generate_batch(model, tokenizer, chunk_prompts, args)
            for i, (record, completion) in enumerate(zip(chunk, completions)):
                _, verdict = grade(
                    completion,
                    record["family"],
                    json.loads(record["solution_json"]),
                    json.loads(record["params_json"]),
                )
                per_record[i].append(verdict.accepted)
                per_record_invalid[i] += int(verdict.invalid)

        for record, results, bad in zip(chunk, per_record, per_record_invalid):
            key = (record["family"], record["difficulty"])
            total[key] += 1
            solved_any[key] += int(any(results))
            solved_first[key] += int(results[0])
            invalid[key] += bad
            detail.append(
                {
                    "task_id": record["task_id"],
                    "family": record["family"],
                    "difficulty": record["difficulty"],
                    "pass_at_1": bool(results[0]),
                    "pass_at_k": bool(any(results)),
                }
            )
        print(f"  {min(start + args.batch_size, len(records))}/{len(records)} instances", flush=True)

    rows = []
    for key in sorted(total):
        family, difficulty = key
        n = total[key]
        rows.append(
            {
                "family": family,
                "difficulty": difficulty,
                "n": n,
                "pass@1": solved_first[key] / n,
                f"pass@{args.samples}": solved_any[key] / n,
                "invalid_rate": invalid[key] / (n * args.samples),
            }
        )

    width = max(len(r["family"]) for r in rows)
    print(f"\n{'family':<{width}}  d  {'n':>4}  {'pass@1':>7}  {'pass@k':>7}  {'invalid':>7}")
    for r in rows:
        print(
            f"{r['family']:<{width}}  {r['difficulty']}  {r['n']:>4}  "
            f"{r['pass@1']:>7.3f}  {r[f'pass@{args.samples}']:>7.3f}  {r['invalid_rate']:>7.3f}"
        )
    overall = sum(solved_first.values()) / max(1, sum(total.values()))
    print(f"\noverall pass@1: {overall:.3f} over {sum(total.values())} instances")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {"model": args.model, "adapter": args.adapter, "k": args.samples,
                 "rows": rows, "detail": detail},
                handle,
                indent=2,
            )
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
