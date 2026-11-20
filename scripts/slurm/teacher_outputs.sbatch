#!/usr/bin/env bash
#SBATCH --job-name=medres-teacher
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
# A teacher's outputs on the training prompts of a task, one shard per array task (one GPU each). Everything after
# the script name goes to `medres eval generate`; the shard and the output file are added here:
#
#   sbatch -p <partition> scripts/slurm/teacher_outputs.sbatch --task cxr_report --model models/UniRG-CXR \
#       --data data/cxr_report/train.jsonl
#   # when all shards are done:
#   medres data teacher-outputs --prompts data/cxr_report/train.jsonl --generations gen/shard*.jsonl \
#       --out data/cxr_report/train_teacher.jsonl
#
# GEN (default gen) is the directory of the shard files. With several teachers, submit once per teacher with
# `--tag <teacher_tag>` and its own GEN (`sbatch --export=ALL,GEN=gen/CT ... --tag CT`).
set -euo pipefail
export PATH="${MEDRES_ENV:?set MEDRES_ENV to the directory of the training environment}/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
N=$((SLURM_ARRAY_TASK_MAX + 1))
OUT="${GEN:-gen}/shard${SLURM_ARRAY_TASK_ID}of$N.jsonl"
echo "[$(date)] $(hostname) job=$SLURM_JOB_ID shard $SLURM_ARRAY_TASK_ID/$N -> $OUT"
medres eval generate --profile teacher_outputs --shard "$SLURM_ARRAY_TASK_ID/$N" --out "$OUT" "$@"
