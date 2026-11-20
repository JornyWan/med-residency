#!/usr/bin/env bash
# Download the public models and source datasets of the built-in tasks, most of them from the Hugging Face hub.
#
#   bash scripts/download_models.sh cxr_report            # one or several of the groups below
#   bash scripts/download_models.sh model Qwen/Qwen3-VL-4B-Instruct
#
#   cxr_report        Qwen3-VL-2B (student), UniRG-CXR (teacher), ReXrank test definitions, IU X-ray images
#   med_vqa           Qwen3-VL-2B (student), MediX-R1-8B (teacher), PMC-VQA, MedXpertQA, MMMU (health subjects)
#   modality_experts  Qwen2-VL-2B (student), the eight Med-R1 experts, Med-R1's splits, OmniMedVQA (open-access part)
#   green             the GREEN judge of the report metrics (7B), into the Hugging Face cache
#   model <hub id>    any other model of configs/models.yaml
#
# Models go to $MEDRES_MODELS (default ./models), source datasets to $MEDRES_RAW (default ./raw); `medres data ...`
# turns the sources into the task files under $MEDRES_DATA (docs/data.md).
#
# NOT downloaded here: MIMIC-CXR and MIMIC-CXR-JPG. They are credentialed; get them from PhysioNet under your own
# data-use agreement (docs/data.md). ReXRank_MIMICCXR_test.json, fetched below, contains MIMIC-CXR report text and
# falls under the same agreement.
#
# Every model and dataset has its own licence and terms of use; read them on the hub page before downloading.
set -euo pipefail

MODELS="${MEDRES_MODELS:-models}"
RAW="${MEDRES_RAW:-raw}"
REXRANK=https://raw.githubusercontent.com/rajpurkarlab/ReXrank/main/data
MED_R1=https://github.com/Yuxiang-Lai117/Med-R1

model() {  # hub id [extra `hf download` arguments]
    hf download "$1" --local-dir "$MODELS/$(basename "$1")" "${@:2}"
}

dataset() {  # hub id, directory under $RAW, [files / extra arguments]
    hf download "$1" --repo-type dataset --local-dir "$RAW/$2" "${@:3}"
}

extract() {  # zip file, target directory (python: compute nodes often have no `unzip`)
    python -I -m zipfile -e "$1" "$2"
}

cxr_report() {
    model Qwen/Qwen3-VL-2B-Instruct
    model microsoft/UniRG-CXR
    mkdir -p "$RAW/rexrank"
    curl -sfL "$REXRANK/mimic-cxr/ReXRank_MIMICCXR_test.json" -o "$RAW/rexrank/ReXRank_MIMICCXR_test.json"
    curl -sfL "$REXRANK/iu_xray/ReXRank_IUXray_test.json" -o "$RAW/rexrank/ReXRank_IUXray_test.json"
    dataset dz-osamu/IU-Xray iu_xray image.zip
    extract "$RAW/iu_xray/image.zip" "$RAW/iu_xray/extracted"
    echo "IU X-ray images (the --images of \`medres data cxr-iu\`):"
    dirname "$(find "$RAW/iu_xray/extracted" -type d -name 'CXR1_*' | head -1)"
}

med_vqa() {
    model Qwen/Qwen3-VL-2B-Instruct
    model MBZUAI/MediX-R1-8B
    dataset RadGenome/PMC-VQA pmc_vqa train_2.csv test_2.csv images_2.zip
    extract "$RAW/pmc_vqa/images_2.zip" "$RAW/pmc_vqa"
    dataset TsinghuaC3I/MedXpertQA medxpertqa
    extract "$RAW/medxpertqa/images.zip" "$RAW/medxpertqa/images"
    dataset MMMU/MMMU mmmu --include "Basic_Medical_Science/validation-*" "Clinical_Medicine/validation-*" \
        "Diagnostics_and_Laboratory_Medicine/validation-*" "Pharmacy/validation-*" "Public_Health/validation-*"
}

modality_experts() {
    model Qwen/Qwen2-VL-2B-Instruct
    model yuxianglai117/Med-R1 --include "README.md" "VQA_CT/*" "VQA_MRI/*" "VQA_X-Ray/*" "VQA_Fundus/*" \
        "VQA_Dermo/*" "VQA_Micro/*" "VQA_OCT/*" "VQA_US/*"
    # the training splits are in the code repository only; the hub copy has the test half
    [ -d "$RAW/med_r1" ] || git clone --depth 1 "$MED_R1" "$RAW/med_r1"
    dataset foreverbeliever/OmniMedVQA omnimedvqa
    extract "$RAW/omnimedvqa/OmniMedVQA.zip" "$RAW/omnimedvqa"
}

green() {
    hf download StanfordAIMI/GREEN-RadLlama2-7b
}

[ $# -gt 0 ] || { sed -n '2,20p' "$0"; exit 1; }
while [ $# -gt 0 ]; do
    case "$1" in
        cxr_report | med_vqa | modality_experts | green) "$1" ;;
        model) model "${2:?model needs a hub id}"; shift ;;
        *) echo "unknown group: $1" >&2; exit 1 ;;
    esac
    shift
done
