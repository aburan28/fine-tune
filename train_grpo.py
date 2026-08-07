"""GRPO fine-tuning of a Qwen reasoning model on the cryptanalysis curriculum.

    python train_grpo.py --config configs/qwen3-4b-24gb.yaml

Needs a CUDA GPU. Everything under ``cryptorl/`` except this file and
``evaluate.py`` runs without one, which is how the reward pipeline gets tested.

Why GRPO and not PPO: the reward here is a pinned verifier, so a learned value
head would be a second, worse estimate of something already known exactly.
GRPO's group-relative advantage -- sample ``num_generations`` rollouts per
ciphertext, centre the rewards within the group -- gets the same variance
reduction with no critic and no critic-sized chunk of VRAM.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import yaml

from cryptorl.dataset import build_splits
from cryptorl.rewards import RewardWeights, build_reward_functions
from cryptorl.tasks import CLASSICAL_FAMILIES, FAMILIES, MAX_DIFFICULTY

LOG = logging.getLogger("train_grpo")


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def expand_families(names: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        out.extend(CLASSICAL_FAMILIES if name == "classical" else [name])
    unknown = set(out) - set(FAMILIES)
    if unknown:
        raise SystemExit(f"unknown families in config: {sorted(unknown)}")
    return tuple(out)


def supported_kwargs(config_cls, values: dict) -> dict:
    """Drop config keys the installed TRL does not have.

    ``GRPOConfig`` gains and renames fields between TRL releases (vLLM
    integration and loss-type flags have moved more than once). Failing the run
    on an unknown key would mean this file only works against one pinned
    version; silently dropping one would hide a typo. So: drop, and say so.
    """
    known = {f.name for f in dataclasses.fields(config_cls)}
    accepted = {k: v for k, v in values.items() if k in known}
    rejected = sorted(set(values) - known)
    if rejected:
        LOG.warning(
            "installed %s does not accept %s -- ignoring. Check the TRL version "
            "if you meant to set them.",
            config_cls.__name__,
            ", ".join(rejected),
        )
    return accepted


def render_prompts(records: list[dict], tokenizer) -> list[dict]:
    """Apply the chat template ourselves so thinking is provably on.

    TRL will template a conversational ``prompt`` column for us, but it calls
    ``apply_chat_template`` without ``enable_thinking``. Qwen3 defaults that to
    true today; pinning it here means a template default change cannot silently
    turn the model's reasoning off mid-project, which would look like a reward
    collapse with no code change to blame.
    """
    out = []
    for record in records:
        rendered = tokenizer.apply_chat_template(
            record["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        out.append({**record, "prompt": rendered})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None, help="overrides the config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    # Imported late so `--help` and a config typo do not cost a torch import.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = dict(cfg["training"])
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir

    weights = RewardWeights(**cfg.get("rewards", {}))
    weights.check()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    families = expand_families(data_cfg.get("families", list(FAMILIES)))
    difficulties = tuple(range(data_cfg.get("max_difficulty", MAX_DIFFICULTY) + 1))
    train_records, eval_records = build_splits(
        data_cfg["train_size"],
        data_cfg["eval_size"],
        families=families,
        difficulties=difficulties,
    )
    LOG.info(
        "%d train / %d eval instances over %d families, difficulties %s",
        len(train_records),
        len(eval_records),
        len(families),
        list(difficulties),
    )

    train_ds = Dataset.from_list(render_prompts(train_records, tokenizer))
    eval_ds = Dataset.from_list(render_prompts(eval_records, tokenizer))

    quantization = None
    if model_cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        quantization_config=quantization,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    grpo_config = GRPOConfig(**supported_kwargs(GRPOConfig, train_cfg))
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=build_reward_functions(weights),
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()

    output_dir = Path(grpo_config.output_dir)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    (output_dir / "final" / "cryptorl.json").write_text(
        json.dumps(
            {
                "base_model": model_cfg["name"],
                "families": list(families),
                "difficulties": list(difficulties),
                "reward_weights": dataclasses.asdict(weights),
            },
            indent=2,
        )
    )
    LOG.info("adapter written to %s", output_dir / "final")


if __name__ == "__main__":
    main()
