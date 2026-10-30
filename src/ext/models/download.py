"""Download pinned HF snapshots for the extra models into HF_HOME=~/rs3d/.hf (no login, no gated repos)."""
import sys
from huggingface_hub import snapshot_download

PINS = {
    "depth-anything/DA3-GIANT-1.1": "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": "b2359bdf726fb44ef62acca04d629dcf158053e7",
    "lch01/StreamVGGT": "f9ba55b1955bc4f34337c51142158a9aa2862c7f",
}
for rid in (sys.argv[1:] or PINS):
    p = snapshot_download(rid, revision=PINS[rid])
    print(rid, PINS[rid], p, flush=True)
