#!/usr/bin/env bash
#SBATCH --job-name=medres-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Evaluate one checkpoint on the test sets of a task (generation with vLLM, then scoring). Everything after the script
# name goes to `medres eval run`:
#
#   sbatch -p <partition> scripts/slurm/eval.sbatch --task cxr_report --model runs/cxr_report_adaopd_s42/checkpoint-300
#   sbatch -p <partition> scripts/slurm/eval.sbatch --task cxr_report --model runs/... --green   # also GREEN (7B judge)
#
# Report tasks need MEDRES_METRICS_PYTHON (scripts/install_metrics_env.sh). Evaluate the checkpoints you compare on
# the same GPU type: scores of one checkpoint differ slightly between GPU types.
set -euo pipefail
export PATH="${MEDRES_ENV:?set MEDRES_ENV to the directory of the training environment}/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] $(hostname) job=$SLURM_JOB_ID medres eval run $*"
medres eval run "$@"
