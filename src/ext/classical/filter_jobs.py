"""One-off (2026-10-29): drop SfM job records hit by the SiftGPU CUDA-context failure from jobs.jsonl so they rerun.
Keeps records with valid features AND both mappers 16/16; windows with valid features but < 16 registered are
re-run too (their job may have overlapped the failure); their old recon dirs move to _gpuerr_suspect/.
The unfiltered file is kept as jobs_raw_run2_gpu_with_cuda_errors.jsonl."""
import json
import os
import shutil

import numpy as np

os.chdir(os.path.expanduser("~/rs3d/results/exp2/baselines/classical"))
J = [json.loads(line) for line in open("jobs_raw_run2_gpu_with_cuda_errors.jsonl")]
keep, sus = [], []
for j in J:
    k = j.get("n_kp_median")
    if j["status"] == "ok" and k is not None and np.isfinite(k) and k > 0:
        (keep if all(j["mappers"][m]["n_reg"] == 16 for m in ("colmap", "glomap")) else sus).append(j)
print(len(J), "keep", len(keep), "suspect (rerun)", len(sus), "corrupt (rerun)", len(J) - len(keep) - len(sus))
with open("jobs.jsonl", "w") as f:
    for j in keep:
        f.write(json.dumps(j) + "\n")
for j in sus:
    for m in ("colmap", "glomap"):
        src = "%s/recon/seq%d/%04d_%s" % (m, j["seq"], j["window_id"], j["cond"])
        if os.path.isdir(src):
            dst = "_gpuerr_suspect/" + src
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
