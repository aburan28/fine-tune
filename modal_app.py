"""Run the cryptanalysis GRPO fine-tune on Modal.

    pip install -r requirements-modal.txt
    modal setup                                          # one-time device auth
    modal run --detach modal_app.py --config configs/smoke.yaml

Only the `modal` client runs on your machine. Everything below (torch, trl,
transformers, ...) is declared in the `image` object further down and only
ever runs inside the remote container.

Modal is the shorter path than [`aws/`](aws/README.md) for one specific reason:
there is no GPU quota to request. An AWS account that has never run a GPU
instance has a G-family spot limit of zero vCPUs and the first launch fails
with `MaxSpotInstanceCountExceeded`, which is a support ticket and a wait.
Modal bills per second and starts immediately.

The persistence design is the same as the EC2 one, for the same reason: a
container can go away mid-run, so checkpoints, the Hugging Face cache and the
outputs live on a Modal Volume that outlives it, and `train_grpo.py` resumes
from the newest *complete* checkpoint. Re-running this file resumes; it does
not start over.

GPU and timeout are read from the environment rather than taken as arguments,
because Modal fixes both when the function is declared, before any argument
this file's entrypoint receives exists:

    CRYPTORL_GPU=L40S modal run --detach modal_app.py --config configs/qwen3-4b-g7.yaml

`--detach` matters for anything longer than a smoke test: without it the run
dies when your terminal does.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import modal

# A10G is 24GB, which is what configs/qwen3-4b-24gb.yaml is sized for. L40S
# (48GB) is the one to reach for if you want bf16 instead of 4-bit or an 8B
# base -- see configs/qwen3-4b-g7.yaml and configs/qwen3-8b-g7e.yaml, whose
# VRAM assumptions carry over even though the card names do not.
GPU = os.environ.get("CRYPTORL_GPU", "A10G")
TIMEOUT_HOURS = float(os.environ.get("CRYPTORL_HOURS", "12"))

# Named, so a later `modal run` finds the same checkpoints. Deleting it with
# `modal volume delete cryptorl` is what "start over" means.
VOLUME = modal.Volume.from_name("cryptorl", create_if_missing=True)
VOL = "/vol"
APP_DIR = "/root/app"

REPO = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.51",
        "trl>=0.14",
        "peft>=0.13",
        "datasets>=3.0",
        "accelerate>=1.0",
        "PyYAML>=6.0",
        "bitsandbytes",
    )
    # Your working copy, not a git clone: the point of iterating here is to run
    # the reward function you just edited, and a clone would silently run main.
    .add_local_dir(
        REPO,
        APP_DIR,
        # node_modules: an unrelated npm-managed tool has, at least once, ended
        # up installed in this directory. add_local_dir reads the filesystem, not
        # git, so untracking it (see the PR that did that) does not stop it from
        # being synced into the image -- only excluding it here does.
        ignore=[".git", "artifacts", "runs", "data", "__pycache__", ".pytest_cache",
                "node_modules", "package.json", "package-lock.json"],
    )
)

app = modal.App("cryptanalysis-grpo", image=image)

# Only attached when asked for. Qwen3 is not gated, so the common case needs no
# token, and naming a secret that does not exist fails the whole run at startup.
SECRETS = (
    [modal.Secret.from_name(os.environ["CRYPTORL_HF_SECRET"])]
    if os.environ.get("CRYPTORL_HF_SECRET")
    else []
)


def _commit_periodically(stop: threading.Event, seconds: int = 300) -> None:
    """Push the volume's contents up while training is still running.

    Without this a container that dies takes every checkpoint since the last
    commit with it. Committing on a timer can capture a checkpoint directory
    mid-write, which is exactly the case `latest_complete_checkpoint` in
    train_grpo.py skips -- the two are meant to be read together.
    """
    while not stop.wait(seconds):
        try:
            VOLUME.commit()
        except Exception as exc:  # a failed commit must not kill the training
            print(f"[commit] failed, will retry in {seconds}s: {exc}", flush=True)


@app.function(
    gpu=GPU,
    volumes={VOL: VOLUME},
    timeout=int(TIMEOUT_HOURS * 3600),
    secrets=SECRETS,
    # Modal reclaims containers. A retry re-enters this function, which finds
    # the volume's checkpoints and continues; without the resume logic it would
    # cheerfully start from step 0 three times.
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
)
def run(config: str, eval_size: int, samples: int, max_difficulty: int) -> dict:
    env = dict(
        os.environ,
        HF_HOME=f"{VOL}/hf",
        TOKENIZERS_PARALLELISM="false",
    )
    runs = Path(f"{VOL}/runs/current")
    state = Path(f"{VOL}/state")
    state.mkdir(parents=True, exist_ok=True)

    model = _model_from_config(Path(APP_DIR) / config)
    print(f"gpu={GPU} config={config} model={model}", flush=True)
    subprocess.run(["nvidia-smi"], check=False)

    stop = threading.Event()
    committer = threading.Thread(target=_commit_periodically, args=(stop,), daemon=True)
    committer.start()

    def sh(*argv: str) -> None:
        print(f"\n$ {' '.join(argv)}\n", flush=True)
        subprocess.run(argv, cwd=APP_DIR, env=env, check=True)

    try:
        # Keyed on a marker, not on "is this the first attempt": after a retry
        # it is not the first attempt but the baseline is still already done.
        if not (state / "baseline.done").exists():
            sh("python", "evaluate.py", "--model", model, "--size", str(eval_size),
               "--samples", str(samples), "--max-difficulty", str(max_difficulty),
               "--out", f"{VOL}/baseline.json")
            (state / "baseline.done").touch()
            VOLUME.commit()

        sh("python", "train_grpo.py", "--config", config, "--output-dir", str(runs))

        sh("python", "evaluate.py", "--model", model, "--adapter", str(runs / "final"),
           "--size", str(eval_size), "--samples", str(samples),
           "--max-difficulty", str(max_difficulty), "--out", f"{VOL}/tuned.json")
    finally:
        stop.set()
        committer.join(timeout=30)
        VOLUME.commit()

    return {
        "baseline": _read_json(Path(f"{VOL}/baseline.json")),
        "tuned": _read_json(Path(f"{VOL}/tuned.json")),
        "adapter": str(runs / "final"),
    }


def _model_from_config(path: Path) -> str:
    """The base model comes from the config, never a default.

    evaluate.py takes --model and train_grpo.py reads it from the config. Left
    to drift they will disagree, and the failure lands at the very end: the
    tuned evaluation tries to load the adapter onto a different base and dies
    after the GPU time is already spent.
    """
    import yaml

    with path.open() as handle:
        return yaml.safe_load(handle)["model"]["name"]


def _read_json(path: Path):
    import json

    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


@app.local_entrypoint()
def main(
    config: str = "configs/qwen3-4b-24gb.yaml",
    eval_size: int = 128,
    samples: int = 4,
    max_difficulty: int = 1,
):
    print(f"gpu={GPU}  timeout={TIMEOUT_HOURS}h  config={config}")
    result = run.remote(config, eval_size, samples, max_difficulty)

    for label in ("baseline", "tuned"):
        report = result.get(label)
        if not report:
            print(f"\n{label}: not produced")
            continue
        print(f"\n{label}:")
        for row in report["rows"]:
            print(f"  {row['family']:<20} d{row['difficulty']}  "
                  f"pass@1 {row['pass@1']:.3f}  invalid {row['invalid_rate']:.3f}")

    print(f"\nadapter is on the volume at {result['adapter']}")
    print("fetch it with:  modal volume get cryptorl runs/current/final ./artifacts")
