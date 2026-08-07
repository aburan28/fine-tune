#!/bin/bash
# Runs as user-data on every launch. `run-on-ec2.sh up` prepends /etc/finetune/env.
#
# This must be idempotent and it must be safe to run on an instance whose data
# volume already holds a half-finished run: that is the normal case after a spot
# interruption, not an edge case. The single most destructive thing this script
# could do is reformat a volume that already has checkpoints on it, so the
# filesystem check below refuses to mkfs anything carrying a filesystem.
set -euxo pipefail

# shellcheck disable=SC1091
source /etc/finetune/env

MNT=/mnt/ft
STATE="$MNT/state"
LOGS="$MNT/logs"

phase () { mkdir -p "$STATE"; echo "$1" >"$STATE/phase"; echo "=== phase: $1 ==="; }

# --- hard cost ceiling -------------------------------------------------------
# Scheduled before anything that can hang. The instance is launched with
# instance-initiated-shutdown-behavior=terminate, so this halts billing even if
# training wedges, the network dies, or a pip resolver spins forever. The data
# volume is a separate attached volume and survives the termination.
shutdown -h "+$((FT_MAX_HOURS * 60))" &
disown || true

phase bootstrap

# --- find and mount the data volume -----------------------------------------
# AWS exposes the volume id as the NVMe serial, so by-id is exact. Guessing at
# /dev/nvme1n1 would eventually pick the root disk on some instance type and
# the mkfs below would destroy it.
DEV=""
for _ in $(seq 1 60); do
  for candidate in /dev/disk/by-id/*"${FT_VOLUME_ID//-/}"*; do
    if [ -b "$candidate" ]; then DEV="$(readlink -f "$candidate")"; break 2; fi
  done
  sleep 5
done
if [ -z "$DEV" ]; then
  echo "FATAL: data volume $FT_VOLUME_ID never appeared" >&2
  phase failed-no-volume
  exit 1
fi

# blkid succeeds only if the device already carries a filesystem. Formatting is
# therefore reachable exactly once, on a volume this script has never seen.
if ! blkid "$DEV"; then
  echo "no filesystem on $DEV -- formatting (first launch)"
  mkfs.ext4 -m 0 "$DEV"
else
  echo "$DEV already has a filesystem -- mounting without touching it"
fi
mkdir -p "$MNT"
mount "$DEV" "$MNT"
mkdir -p "$STATE" "$LOGS" "$MNT/hf" "$MNT/runs"

# --- code --------------------------------------------------------------------
phase fetching-code
if [ -d "$MNT/repo/.git" ]; then
  git -C "$MNT/repo" fetch --depth 1 origin "$FT_BRANCH"
  git -C "$MNT/repo" reset --hard "origin/$FT_BRANCH"
else
  git clone --depth 1 --branch "$FT_BRANCH" "$FT_REPO" "$MNT/repo"
fi

# --- python ------------------------------------------------------------------
# The Deep Learning AMI ships torch and the NVIDIA driver already; reinstalling
# torch here would pull several GB and risk a CUDA/driver mismatch. Find its
# interpreter rather than assuming a path, since the layout has moved between
# DLAMI releases.
phase installing-deps
PY=""
for candidate in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do
  if [ -x "$candidate" ] && "$candidate" -c "import torch" 2>/dev/null; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
  echo "no preinstalled torch found; falling back to pip (slow)" >&2
  PY=/usr/bin/python3
  "$PY" -m pip install --break-system-packages torch
fi
"$PY" -m pip install --break-system-packages -q \
  "transformers>=4.51" "trl>=0.14" "peft>=0.13" "datasets>=3.0" "accelerate>=1.0" "PyYAML>=6.0" bitsandbytes
echo "$PY" >"$STATE/python"

# --- the run -----------------------------------------------------------------
cat >"$MNT/run.sh" <<'RUNNER'
#!/bin/bash
set -euxo pipefail
source /etc/finetune/env
MNT=/mnt/ft; STATE="$MNT/state"; LOGS="$MNT/logs"
PY="$(cat "$STATE/python")"
cd "$MNT/repo"
export HF_HOME="$MNT/hf"          # survives interruption; re-downloading 4B every launch is the single biggest waste
export TOKENIZERS_PARALLELISM=false
phase () { echo "$1" >"$STATE/phase"; }

# The baseline is only meaningful before training and only needs doing once, so
# it is keyed on a marker rather than on "is this the first boot" -- after an
# interruption it is not the first boot but the baseline is still done.
if [ ! -f "$STATE/baseline.done" ]; then
  phase baseline-eval
  "$PY" evaluate.py --model "$FT_MODEL" --size "$FT_EVAL_SIZE" --samples "$FT_EVAL_SAMPLES" \
    --max-difficulty "$FT_MAX_DIFFICULTY" --out "$MNT/baseline.json" 2>&1 | tee -a "$LOGS/baseline.log"
  touch "$STATE/baseline.done"
fi

phase training
"$PY" train_grpo.py --config "$FT_CONFIG" --output-dir "$MNT/runs/current" 2>&1 | tee -a "$LOGS/train.log"

phase tuned-eval
"$PY" evaluate.py --model "$FT_MODEL" --adapter "$MNT/runs/current/final" \
  --size "$FT_EVAL_SIZE" --samples "$FT_EVAL_SAMPLES" --max-difficulty "$FT_MAX_DIFFICULTY" \
  --out "$MNT/tuned.json" 2>&1 | tee -a "$LOGS/eval.log"

phase done
touch "$STATE/run.done"
sync

# Stop paying for a GPU that has nothing left to do. `fetch` can still pull the
# adapter afterwards: it lives on the data volume, which outlives the instance.
if [ "$FT_SHUTDOWN_WHEN_DONE" = "1" ]; then shutdown -h now; fi
RUNNER
chmod +x "$MNT/run.sh"

cat >/etc/systemd/system/finetune.service <<'UNIT'
[Unit]
Description=cryptanalysis GRPO run
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/mnt/ft/run.sh
RemainAfterExit=yes
StandardOutput=append:/mnt/ft/logs/run.log
StandardError=append:/mnt/ft/logs/run.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now finetune.service
