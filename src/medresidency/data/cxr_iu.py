"""ReXrank's IU X-ray test set -> `cxr_report/test_iu.jsonl`, an out-of-distribution test of the `cxr_report` task.

Same prompt, image size and reference format as the MIMIC-CXR files (`medres data cxr-report`); IU X-ray is public,
so this file can be built without credentialed access. Sources (docs/data.md): ReXrank's ReXRank_IUXray_test.json
and the IU X-ray images in the layout its `key_image_path` refers to.
"""

from __future__ import annotations

import argparse
import json
import os
import re

from medresidency import config, jsonl
from medresidency.data.cxr_report import SIZE, clean, eval_row, rexrank_context, section_counts


def main():
    p = argparse.ArgumentParser(
        prog="medres data cxr-iu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rexrank", required=True, help="ReXRank_IUXray_test.json")
    p.add_argument("--images", required=True, help="directory the key_image_path entries are relative to")
    p.add_argument("--out", default=config.data_path("cxr_report"))
    a = p.parse_args()
    from PIL import Image

    os.makedirs(f"{a.out}/img_iu", exist_ok=True)
    with open(a.rexrank) as f:
        rex = json.load(f)
    rows, missing = [], 0
    for sid, r in rex.items():
        src = os.path.join(a.images, r["key_image_path"].lstrip("/"))
        if not os.path.exists(src):
            missing += 1
            continue
        dst = f"{a.out}/img_iu/" + re.sub(r"[^A-Za-z0-9_.-]", "_", str(sid)) + ".png"
        if not os.path.exists(dst):
            Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(dst)
        row = {
            "id": str(sid),
            "img": dst,
            "context": rexrank_context(r.get("context")),
            "findings": clean(r.get("section_findings")),
            "impression": clean(r.get("section_impression")),
        }
        rows.append(row)
    print(f"ReXrank IU X-ray test: {len(rex)} studies, {missing} without their image; {section_counts(rows)}")
    print("test_iu.jsonl", jsonl.write(f"{a.out}/test_iu.jsonl", map(eval_row, rows)))


if __name__ == "__main__":
    main()
