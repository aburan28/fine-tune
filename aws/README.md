# Running the fine-tune on an EC2 spot GPU

```bash
./aws/run-on-ec2.sh price     # what it would cost, launches nothing
./aws/run-on-ec2.sh up        # shows a plan, asks, then launches
./aws/run-on-ec2.sh logs      # follow it
./aws/run-on-ec2.sh fetch     # pull the adapter down
./aws/run-on-ec2.sh down      # stop paying
```

`up` does the whole thing: resolves the current Deep Learning AMI, picks the
cheapest availability zone, creates a key pair and a security group, creates a
data volume, launches a spot instance, and attaches the volume. The instance
then installs, runs a baseline evaluation, trains, runs a second evaluation,
and terminates itself.

## Spot, and being interrupted

Spot is roughly half the on-demand price — `g7.2xlarge` has been running near
$0.68/hr — and AWS can reclaim the instance with two
minutes' notice. A GRPO run is many hours, so **assume it will be interrupted**
and make that cheap rather than trying to avoid it:

- Checkpoints, the Hugging Face cache and the outputs all live on a separate
  EBS volume that is not deleted when the instance dies.
- `train_grpo.py` resumes from the newest checkpoint unless you pass `--fresh`.
- `save_steps: 50` bounds what an interruption costs to 50 steps.

So recovery is just:

```bash
./aws/run-on-ec2.sh up
```

It finds the existing volume, launches into that volume's zone, re-attaches it,
and training picks up from the last save. The base model is not re-downloaded
either — `HF_HOME` is on the volume.

### Why a volume instead of syncing checkpoints to S3

The usual pattern is an IAM instance profile granting S3 access. That needs
`iam:CreateRole` and `iam:PassRole`, which this account's user does not have.
The alternative — putting access keys in user-data — writes long-lived
credentials somewhere every process on the instance can read, and user-data is
retrievable from the metadata service by anything running on the box. The
volume needs no credentials on the instance at all, so that is what this uses.

If you do get an instance profile, syncing to S3 is strictly better: it
removes the availability-zone pin described below.

### The availability-zone trade-off

An EBS volume attaches only within its own zone, so once the volume exists the
zone is fixed and spot capacity is drawn from that one zone. That is a real
reduction in flexibility, accepted deliberately: the alternative is discarding
the checkpoints the volume holds. `down --delete-volume` and a fresh `up` will
pick a new cheapest zone.

## What it costs

| item | VRAM | spot rate | note |
|---|---|---|---|
| **`g7.2xlarge`** | 32GB | ~$0.68/hr | RTX PRO 4500. **The default** |
| `g5.2xlarge` | 24GB | ~$0.63/hr | A10G. Older, and needs 4-bit |
| `g7.4xlarge` | 32GB | ~$0.91/hr | same GPU, 16 vCPU |
| `g7e.2xlarge` | 96GB | ~$1.58/hr | RTX PRO Server 6000. 8B base + colocated vLLM |
| `g6e.xlarge` | 45GB | ~$1.46/hr | L40S |
| 200GB gp3 volume | — | ~$0.53/day | **keeps billing after the instance is gone** |

Rates move; `price` reads them live. `g7.2xlarge` is the default because 32GB
costs within a nickel of the 24GB `g5.2xlarge`, and that extra headroom is what
lets [`configs/qwen3-4b-g7.yaml`](../configs/qwen3-4b-g7.yaml) run the 4B model
in bf16 instead of 4-bit NF4 — quantization costs accuracy on exactly the long
arithmetic chains these tasks are built from.

For the harder families, `g7e.2xlarge`:

```bash
./aws/run-on-ec2.sh up --instance-type g7e.2xlarge \
    --config configs/qwen3-8b-g7e.yaml --model Qwen/Qwen3-8B
```

96GB buys an 8B base — 4B mostly cannot hold a continued-fraction argument
together, so `rsa-wiener` and the deeper `dlp-secp256k1` tiers never earn reward
and contribute only variance — and enough headroom for vLLM colocated with
training, which is the difference between generation taking a night and taking
a weekend.

Three things stop a runaway bill:

1. `--max-hours` (default 12) schedules `shutdown -h` at boot. The instance is
   launched with `instance-initiated-shutdown-behavior=terminate`, so that halts
   billing even if training hangs or the network dies.
2. The run terminates the instance itself when it finishes.
3. `up` refuses to launch if an instance already exists.

The volume is the one thing that keeps costing money silently. `down` reminds
you it is there; `down --delete-volume` removes it and everything on it.

## Failure you will most likely hit first

**`MaxSpotInstanceCountExceeded` / `VcpuLimitExceeded`.** New AWS accounts get a
quota of **0** vCPUs for G-family spot instances, so the very first `up` fails
at `run-instances`. It is a quota request, not a billing problem:

Service Quotas → EC2 → *All G and VT Spot Instance Requests* → request at least
8 vCPUs (a `g7.2xlarge` is 8). Approval is usually hours, sometimes a day. The
on-demand equivalent is a separate quota, *Running On-Demand G and VT
instances*.

This account's user cannot read Service Quotas, so `up` cannot check your quota
before launching — it will surface the error from `run-instances` instead. The
on-demand quota is separate and often nonzero when the spot one is not, so
`--on-demand` is worth trying before waiting on a quota request.

**No capacity in the zone.** `InsufficientInstanceCapacity` means spot has
nothing in that zone right now. Try `--instance-type g5.2xlarge`, or another
region with `--region`. Both `g7` and `g7e` are offered in all four us-west-2
zones, so capacity is usually findable.

**A driver or CUDA mismatch on g7/g7e.** These are a new GPU generation, so the
AMI has to be recent enough to have CUDA support for the card. `up` selects the
*highest PyTorch version* of the Deep Learning AMI rather than the most recently
built one — sorting by build date alone returns a PyTorch 2.10 image while 2.12
exists. Override with `--ami ami-...` if you need a specific one.

**SSH times out.** Your IP changed. Every subcommand re-authorizes the current
address, so run it again. The security group only ever allows port 22 from the
address that launched it, never `0.0.0.0/0`.

## Options

```
--instance-type g7e.2xlarge  default g7.2xlarge
--ami ami-0123...            override the Deep Learning AMI choice
--region us-east-1           default from your aws config
--config configs/smoke.yaml  run the 10-minute smoke on a GPU instead
--max-hours 6                hard ceiling before self-termination
--max-price 0.80             bid cap; raises interruption risk if the market exceeds it
--volume-size 400            default 200GB
--on-demand                  skip spot: ~2x the price, cannot be interrupted
--keep-alive                 do not self-terminate when the run finishes
-y                           skip the confirmation prompt
```

## What has and has not been tested

The launcher's read-only paths were run against real AWS: AMI resolution, zone
selection, spot pricing, and the full `up` preflight through to the
confirmation prompt. The mutating path — key pair, security group, volume,
`run-instances`, attach — was exercised against a stubbed CLI that records
calls, which verifies the ordering and that the generated user-data is valid
bash, but not that AWS accepts every argument.

**No GPU instance has actually been launched.** The bootstrap, the training run
and the evaluations are unrun. Budget the first `up` as a debugging session and
use `--config configs/smoke.yaml --max-hours 1` for it.
