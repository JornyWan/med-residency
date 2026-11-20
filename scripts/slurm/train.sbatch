#!/usr/bin/env bash
#SBATCH --job-name=medres-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# One training run on the GPUs of one node. Everything after the script name goes to `medres train`:
#
#   mkdir -p logs
#   sbatch -p <partition> scripts/slurm/train.sbatch --task cxr_report --method adaopd --seed 42
#   sbatch -p <partition> --gres=gpu:h100:4 scripts/slurm/train.sbatch configs/recipes/adaopd/cxr_report_4b_adaopd.yaml
#
# Submit from the repository root with MEDRES_ENV (the training environment) set; MEDRES_DATA, MEDRES_MODELS and, for
# the report reward of GRPO, MEDRES_METRICS_PYTHON are passed on when set. The request must match `hardware.gpus` of
# the run (default 4: two cards for the student, two for the teacher servers; docs/concepts.md).
#
# A run that was cut short (time limit, preemption) continues from its last checkpoint when the same command is
# submitted again with run.resume=true.
set -euo pipefail
export PATH="${MEDRES_ENV:?set MEDRES_ENV to the directory of the training environment}/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] $(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} medres train $*"
medres train "$@"
