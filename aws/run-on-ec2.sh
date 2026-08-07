#!/usr/bin/env bash
#
# Run the cryptanalysis GRPO fine-tune on an EC2 spot GPU instance, end to end.
#
#   ./aws/run-on-ec2.sh up          launch, bootstrap, train, evaluate
#   ./aws/run-on-ec2.sh status      where the run is
#   ./aws/run-on-ec2.sh logs        follow the training log
#   ./aws/run-on-ec2.sh fetch       copy the adapter and eval reports here
#   ./aws/run-on-ec2.sh down        terminate the instance (billing stops)
#   ./aws/run-on-ec2.sh price       current spot prices, launch nothing
#
# Spot is roughly half the on-demand price and can be reclaimed with two
# minutes' notice, so this is built around being interrupted rather than around
# hoping it is not:
#
#   * Checkpoints, the HF cache and the outputs live on a separate EBS volume
#     that is NOT deleted when the instance dies.
#   * `train_grpo.py` resumes from the newest checkpoint by default.
#   * So recovering from an interruption is `up` again. It re-attaches the same
#     volume and picks up from the last save.
#
# Why the volume rather than syncing checkpoints to S3: giving the instance S3
# access needs an IAM instance profile, and the alternative -- baking access
# keys into user-data -- puts long-lived credentials somewhere every process on
# the box can read. The volume needs no credentials on the instance at all.

set -euo pipefail

REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || echo us-west-2)}"
INSTANCE_TYPE="${FT_INSTANCE_TYPE:-g5.2xlarge}"
CONFIG="configs/qwen3-4b-24gb.yaml"
MODEL="Qwen/Qwen3-4B"
MAX_HOURS=12
VOLUME_SIZE=200
MAX_PRICE=""
REPO="https://github.com/aburan28/fine-tune.git"
BRANCH="main"
EVAL_SIZE=128
EVAL_SAMPLES=4
MAX_DIFFICULTY=1
SHUTDOWN_WHEN_DONE=1
ASSUME_YES=0

TAG_PROJECT="cryptanalysis-finetune"
KEY_NAME="finetune-spot"
SG_NAME="finetune-spot-sg"
VOLUME_TAG="finetune-data"
KEY_FILE="$HOME/.ssh/${KEY_NAME}-${REGION}.pem"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die () { echo "error: $*" >&2; exit 1; }
say () { printf '\033[1m%s\033[0m\n' "$*"; }
aws_ () { aws --region "$REGION" "$@"; }

usage () { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# --- argument parsing --------------------------------------------------------

CMD="${1:-}"; shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --region)         REGION="$2"; KEY_FILE="$HOME/.ssh/${KEY_NAME}-${REGION}.pem"; shift 2 ;;
    --instance-type)  INSTANCE_TYPE="$2"; shift 2 ;;
    --config)         CONFIG="$2"; shift 2 ;;
    --model)          MODEL="$2"; shift 2 ;;
    --max-hours)      MAX_HOURS="$2"; shift 2 ;;
    --volume-size)    VOLUME_SIZE="$2"; shift 2 ;;
    --max-price)      MAX_PRICE="$2"; shift 2 ;;
    --repo)           REPO="$2"; shift 2 ;;
    --branch)         BRANCH="$2"; shift 2 ;;
    --eval-size)      EVAL_SIZE="$2"; shift 2 ;;
    --max-difficulty) MAX_DIFFICULTY="$2"; shift 2 ;;
    --keep-alive)     SHUTDOWN_WHEN_DONE=0; shift ;;
    --delete-volume)  DELETE_VOLUME=1; shift ;;
    -y|--yes)         ASSUME_YES=1; shift ;;
    -h|--help)        usage 0 ;;
    *) die "unknown option $1" ;;
  esac
done

# --- shared helpers ----------------------------------------------------------

require_aws () {
  command -v aws >/dev/null || die "aws CLI not found"
  aws_ sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials are not working for region $REGION"
}

# The AWS CLI renders a null query result as the literal string "None", which
# is truthy in shell and burned an hour the first time it reached a comparison.
# Both lookups below normalise it to empty so callers can just test for that.
none_to_empty () { local v; v="$(cat)"; [ "$v" = "None" ] && echo "" || echo "$v"; }

find_instance () {
  aws_ ec2 describe-instances \
    --filters "Name=tag:Project,Values=$TAG_PROJECT" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[0].InstanceId' --output text 2>/dev/null \
    | tr -d '\n' | none_to_empty
}

instance_ip () {
  aws_ ec2 describe-instances --instance-ids "$1" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

find_volume () {
  aws_ ec2 describe-volumes --filters "Name=tag:Name,Values=$VOLUME_TAG" \
    "Name=status,Values=available,in-use" \
    --query 'Volumes[0].VolumeId' --output text 2>/dev/null | tr -d '\n' | none_to_empty
}

my_ip () { curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]'; }

# SSH is opened to the caller's address only. A GPU box with 22 open to
# 0.0.0.0/0 is found by scanners in minutes, and this one has no reason to
# accept a connection from anywhere else.
ensure_ssh_access () {
  local sg="$1" ip; ip="$(my_ip)/32"
  aws_ ec2 authorize-security-group-ingress --group-id "$sg" \
    --protocol tcp --port 22 --cidr "$ip" >/dev/null 2>&1 || true
  echo "$ip"
}

ssh_ () {
  local ip="$1"; shift
  ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
      -o LogLevel=ERROR "ubuntu@$ip" "$@"
}

# --start-time rather than --max-items: --max-items makes the CLI paginate and
# append a NextToken line to text output, which has no price column, so it
# sorted to the front and every quoted price came back as "None".
spot_price () {
  aws_ ec2 describe-spot-price-history --instance-types "$1" \
    --product-descriptions "Linux/UNIX" \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text 2>/dev/null \
    | awk 'NF == 2'
}

cheapest_az () {
  spot_price "$INSTANCE_TYPE" | sort -k2 -g | head -1 | awk '{print $1}'
}

# --- price -------------------------------------------------------------------

cmd_price () {
  require_aws
  say "spot prices in $REGION"
  for t in g5.xlarge g5.2xlarge g6.2xlarge g6e.xlarge; do
    printf '%-14s' "$t"
    spot_price "$t" | sort -k2 -g | head -1 | awk '{printf "%s  $%s/hr\n", $1, $2}'
  done
  echo
  echo "On-demand g5.2xlarge is about \$1.21/hr for comparison."
}

# --- up ----------------------------------------------------------------------

cmd_up () {
  require_aws

  local existing; existing="$(find_instance)"
  if [ -n "$existing" ]; then
    say "instance $existing already exists ($(instance_ip "$existing"))"
    echo "Use 'status', 'logs', or 'down'. Refusing to launch a second GPU instance."
    exit 0
  fi

  [ -f "$SCRIPT_DIR/bootstrap.sh" ] || die "missing $SCRIPT_DIR/bootstrap.sh"

  say "resolving the Deep Learning AMI"
  local ami
  ami="$(aws_ ec2 describe-images --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*" \
              "Name=state,Values=available" \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)"
  [ -n "$ami" ] && [ "$ami" != "None" ] || die "no Deep Learning AMI found in $REGION"
  echo "  $ami"

  # The volume can only attach to an instance in its own AZ, so an existing
  # volume pins the AZ. That costs some spot flexibility and is worth it: the
  # alternative is losing the checkpoints it holds.
  local volume az
  volume="$(find_volume)"
  if [ -n "$volume" ]; then
    az="$(aws_ ec2 describe-volumes --volume-ids "$volume" \
      --query 'Volumes[0].AvailabilityZone' --output text)"
    say "reusing data volume $volume in $az (a previous run's checkpoints live here)"
  else
    az="$(cheapest_az)"
    [ -n "$az" ] || die "no spot price history for $INSTANCE_TYPE in $REGION"
    say "creating a ${VOLUME_SIZE}GB data volume in $az"
  fi

  local price
  price="$(spot_price "$INSTANCE_TYPE" | sort -k2 -g | head -1 | awk '{print $2}')"

  cat <<PLAN

  region          $REGION
  instance        $INSTANCE_TYPE (spot, one-time)
  availability    $az
  spot price      \$${price}/hr   (max ${MAX_HOURS}h => about \$$(awk "BEGIN{printf \"%.2f\", $price*$MAX_HOURS}"))
  ami             $ami
  data volume     ${volume:-<new>} (${VOLUME_SIZE}GB, survives interruption)
  config          $CONFIG
  model           $MODEL

  Billing starts now and stops when the instance terminates. It terminates by
  itself when the run finishes, or after ${MAX_HOURS}h whichever comes first.
  'down' terminates it early. The data volume keeps costing ~\$$(awk "BEGIN{printf \"%.2f\", $VOLUME_SIZE*0.08/30}")/day
  until 'down --delete-volume'.

PLAN

  if [ "$ASSUME_YES" != "1" ]; then
    read -r -p "Launch? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { echo "aborted"; exit 1; }
  fi

  # key pair
  if ! aws_ ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
    say "creating key pair -> $KEY_FILE"
    mkdir -p "$(dirname "$KEY_FILE")"
    aws_ ec2 create-key-pair --key-name "$KEY_NAME" \
      --query 'KeyMaterial' --output text >"$KEY_FILE"
    chmod 600 "$KEY_FILE"
  fi
  [ -f "$KEY_FILE" ] || die "key pair $KEY_NAME exists in AWS but $KEY_FILE is missing.
Delete the key pair (aws ec2 delete-key-pair --key-name $KEY_NAME --region $REGION) and rerun."

  # security group
  local sg
  sg="$(aws_ ec2 describe-security-groups --group-names "$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
  if [ -z "$sg" ] || [ "$sg" = "None" ]; then
    say "creating security group $SG_NAME"
    sg="$(aws_ ec2 create-security-group --group-name "$SG_NAME" \
      --description "cryptanalysis fine-tune: ssh from the launching host only" \
      --query 'GroupId' --output text)"
  fi
  local cidr; cidr="$(ensure_ssh_access "$sg")"
  echo "  ssh allowed from $cidr"

  # data volume
  if [ -z "$volume" ]; then
    volume="$(aws_ ec2 create-volume --availability-zone "$az" --size "$VOLUME_SIZE" \
      --volume-type gp3 \
      --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=$VOLUME_TAG},{Key=Project,Value=$TAG_PROJECT}]" \
      --query 'VolumeId' --output text)"
    aws_ ec2 wait volume-available --volume-ids "$volume"
    echo "  created $volume"
  fi

  # user-data: the env block the bootstrap sources, then the bootstrap itself
  local udata; udata="$(mktemp)"; trap 'rm -f "$udata"' RETURN
  {
    printf '#!/bin/bash\nset -eux\nmkdir -p /etc/finetune\ncat >/etc/finetune/env <<"FTENV"\n'
    printf 'FT_REPO=%s\n'            "$REPO"
    printf 'FT_BRANCH=%s\n'          "$BRANCH"
    printf 'FT_CONFIG=%s\n'          "$CONFIG"
    printf 'FT_MODEL=%s\n'           "$MODEL"
    printf 'FT_MAX_HOURS=%s\n'       "$MAX_HOURS"
    printf 'FT_VOLUME_ID=%s\n'       "$volume"
    printf 'FT_EVAL_SIZE=%s\n'       "$EVAL_SIZE"
    printf 'FT_EVAL_SAMPLES=%s\n'    "$EVAL_SAMPLES"
    printf 'FT_MAX_DIFFICULTY=%s\n'  "$MAX_DIFFICULTY"
    printf 'FT_SHUTDOWN_WHEN_DONE=%s\n' "$SHUTDOWN_WHEN_DONE"
    printf 'FTENV\n'
    tail -n +2 "$SCRIPT_DIR/bootstrap.sh"
  } >"$udata"

  local subnet
  subnet="$(aws_ ec2 describe-subnets --filters "Name=availability-zone,Values=$az" \
    --query 'Subnets[0].SubnetId' --output text)"
  [ -n "$subnet" ] && [ "$subnet" != "None" ] || die "no subnet in $az"

  local market='MarketType=spot,SpotOptions={SpotInstanceType=one-time}'
  [ -n "$MAX_PRICE" ] && market="MarketType=spot,SpotOptions={SpotInstanceType=one-time,MaxPrice=$MAX_PRICE}"

  say "requesting the spot instance"
  local iid
  iid="$(aws_ ec2 run-instances \
    --image-id "$ami" --instance-type "$INSTANCE_TYPE" --count 1 \
    --key-name "$KEY_NAME" --security-group-ids "$sg" --subnet-id "$subnet" \
    --associate-public-ip-address \
    --instance-market-options "$market" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "file://$udata" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=finetune-spot},{Key=Project,Value=$TAG_PROJECT}]" \
    --query 'Instances[0].InstanceId' --output text)" || die "launch failed -- see the note on quotas in aws/README.md"
  echo "  $iid"

  say "waiting for it to run"
  aws_ ec2 wait instance-running --instance-ids "$iid"

  say "attaching $volume"
  aws_ ec2 attach-volume --volume-id "$volume" --instance-id "$iid" --device /dev/sdf >/dev/null

  local ip; ip="$(instance_ip "$iid")"
  cat <<NEXT

  $iid is up at $ip

  The bootstrap installs, then trains, then evaluates. Give it ~10 minutes
  before the log has anything interesting in it.

    ./aws/run-on-ec2.sh status
    ./aws/run-on-ec2.sh logs
    ./aws/run-on-ec2.sh fetch     # once the phase reads 'done'
    ./aws/run-on-ec2.sh down      # stop paying

  If spot reclaims it, run 'up' again: the volume and its checkpoints outlive
  the instance and training resumes from the last save.

NEXT
}

# --- status / logs / ssh -----------------------------------------------------

with_instance () {
  require_aws
  local iid; iid="$(find_instance)"
  [ -n "$iid" ] || die "no instance found. 'up' to launch one."
  echo "$iid"
}

cmd_status () {
  local iid ip state
  iid="$(with_instance)"
  state="$(aws_ ec2 describe-instances --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"
  ip="$(instance_ip "$iid")"
  say "$iid  $state  $ip"
  [ "$state" = "running" ] || exit 0
  ensure_ssh_access "$(aws_ ec2 describe-security-groups --group-names "$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text)" >/dev/null
  # shellcheck disable=SC2016  # expands on the instance, not here
  ssh_ "$ip" 'echo "phase: $(cat /mnt/ft/state/phase 2>/dev/null || echo starting)";
              nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true;
              tail -3 /mnt/ft/logs/train.log 2>/dev/null || true' \
    || echo "(not reachable yet -- the bootstrap is still running)"
}

cmd_logs () {
  local iid ip; iid="$(with_instance)"; ip="$(instance_ip "$iid")"
  ensure_ssh_access "$(aws_ ec2 describe-security-groups --group-names "$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text)" >/dev/null
  say "following /mnt/ft/logs/run.log (ctrl-c to stop; the run keeps going)"
  ssh_ "$ip" 'tail -F /mnt/ft/logs/run.log /var/log/cloud-init-output.log 2>/dev/null'
}

cmd_ssh () {
  local iid ip; iid="$(with_instance)"; ip="$(instance_ip "$iid")"
  ensure_ssh_access "$(aws_ ec2 describe-security-groups --group-names "$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text)" >/dev/null
  exec ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new "ubuntu@$ip"
}

cmd_fetch () {
  local iid ip; iid="$(with_instance)"; ip="$(instance_ip "$iid")"
  ensure_ssh_access "$(aws_ ec2 describe-security-groups --group-names "$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text)" >/dev/null
  mkdir -p artifacts
  say "copying adapter and reports into ./artifacts"
  scp -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -r \
    "ubuntu@$ip:/mnt/ft/runs/current/final" "ubuntu@$ip:/mnt/ft/baseline.json" \
    "ubuntu@$ip:/mnt/ft/tuned.json" artifacts/ 2>/dev/null \
    || echo "(some artifacts are not there yet -- check 'status')"
  ls -la artifacts/
}

# --- down --------------------------------------------------------------------

cmd_down () {
  require_aws
  local iid; iid="$(find_instance)"
  if [ -n "$iid" ]; then
    say "terminating $iid"
    aws_ ec2 terminate-instances --instance-ids "$iid" \
      --query 'TerminatingInstances[0].CurrentState.Name' --output text
    aws_ ec2 wait instance-terminated --instance-ids "$iid"
    echo "  GPU billing has stopped."
  else
    echo "no instance to terminate"
  fi

  local volume; volume="$(find_volume)"
  if [ "${DELETE_VOLUME:-0}" = "1" ]; then
    if [ -n "$volume" ]; then
      say "deleting data volume $volume -- checkpoints and adapters on it are gone"
      aws_ ec2 wait volume-available --volume-ids "$volume" 2>/dev/null || true
      aws_ ec2 delete-volume --volume-id "$volume"
    fi
  elif [ -n "$volume" ]; then
    echo
    echo "  Data volume $volume kept (~\$$(awk "BEGIN{printf \"%.2f\", $VOLUME_SIZE*0.08/30}")/day)."
    echo "  'up' reuses it and resumes; 'down --delete-volume' removes it for good."
  fi
}

case "$CMD" in
  up)     cmd_up ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  ssh)    cmd_ssh ;;
  fetch)  cmd_fetch ;;
  down)   cmd_down ;;
  price)  cmd_price ;;
  ""|-h|--help) usage 0 ;;
  *) die "unknown command '$CMD' (try --help)" ;;
esac
