#!/usr/bin/env bash
# The report-metric environment: RadEval (RadCliQ-v1, RadGraph-F1, CheXbert, RaTEScore, GREEN).
#
#   bash scripts/install_metrics_env.sh /path/to/envs/medres_metrics
#   export MEDRES_METRICS_PYTHON=/path/to/envs/medres_metrics/bin/python
#
# It is a separate environment because RadEval needs a newer transformers than ms-swift accepts. `medres eval
# report-score`, `medres eval run` (report tasks) and the report reward of GRPO start their scoring process with
# $MEDRES_METRICS_PYTHON; src/medresidency/eval/report_score.py imports nothing from the training environment.
#
# Two patches to the installed RadEval 2.2.1 (both idempotent):
#   1. the package directory is installed as `radeval`, some of its modules import `RadEval.` (fails on Linux);
#   2. with scikit-learn >= 1.8 the CheXbert accuracy line fails (sklearn's sparse count_nonzero gets a dense matrix).
#
# The metric models are downloaded on first use (to the Hugging Face cache; GREEN needs
# StanfordAIMI/GREEN-RadLlama2-7b, see scripts/download_models.sh). Their data directories can be moved with
# STANZA_RESOURCES_DIR, NLTK_DATA and HF_DATASETS_CACHE.
set -euo pipefail

ENV="${1:?usage: install_metrics_env.sh <environment directory>}"
conda create -y -p "$ENV" python=3.12
PY="$ENV/bin/python"
"$PY" -m pip install -U pip
"$PY" -m pip install "RadEval==2.2.1"

P="$("$PY" -c 'import os, importlib.util as u; print(os.path.dirname(u.find_spec("radeval").origin))')"
{ grep -rl -E "(from|import) RadEval\." "$P" --include=*.py || true; } |
    xargs -r sed -i -E 's/\b(from|import) RadEval\./\1 radeval./g'

"$PY" - "$P/metrics/_chexbert_base.py" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = "        pe_accuracy = (count_nonzero(y_true5 - y_pred5, axis=1) == 0).astype(float)\n"
new = ("        _d5 = y_true5 - y_pred5  # medresidency patch: sklearn's sparse count_nonzero rejects a dense matrix\n"
       "        _d5 = _d5.toarray() if hasattr(_d5, 'toarray') else np.asarray(_d5)\n"
       "        pe_accuracy = (np.count_nonzero(_d5, axis=1) == 0).astype(float)\n")
if "medresidency patch" not in src:
    if src.count(old) != 1:
        sys.exit(f"the CheXbert patch did not apply to {path} (another RadEval version?)")
    open(path, "w").write(src.replace(old, new))
EOF

"$PY" -c "from radeval.metrics.RadCliQv1.radcliq import CompositeMetric; print('RadEval ready:', '$P')"
echo "export MEDRES_METRICS_PYTHON=$PY"
