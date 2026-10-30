"""Build ext/vendor_pi3 from the pi3 `training` branch files (verbatim except the marked loss.py edits)."""
import difflib, io

SRC = "github.com/yyfz/Pi3 branch `training` @ 97ae7a43296541859d3c55eae70481f561a6789f"
hdr = lambda f: f'# VENDORED VERBATIM from {SRC}, file {f}\n# (read with `git -C ~/rs3d/third_party/pi3 show origin/training:{f}`)\n'

def rd(p):
    return open(p, encoding="utf-8", newline="").read()

for name, f in [("geometry", "pi3/utils/geometry.py"), ("alignment", "pi3/utils/alignment.py")]:
    open(f"ext/vendor_pi3/{name}.py", "w", encoding="utf-8", newline="").write(hdr(f) + rd(f"ref/pi3_{name}_orig.py"))

orig = rd("ref/pi3_loss_orig.py")
edits = [
    ("from ..utils.geometry import homogenize_points, se3_inverse, depth_edge\nfrom ..utils.alignment import align_points_scale\n",
     "from .geometry import homogenize_points, se3_inverse, depth_edge  # CHANGED: vendored copies\nfrom .alignment import align_points_scale  # CHANGED: vendored copy\n"),
    ("from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__\n",
     "# CHANGED: the repo imports these lists from its `datasets` package. Our Replica renders are clean synthetic data,\n"
     "# i.e. the repo's HIGH_QUALITY class (Hypersim, Habitat, ...), for which the repo applies the normal loss.\n"
     "__HIGH_QUALITY_DATASETS__ = ['ReplicaRS']\n__MIDDLE_QUALITY_DATASETS__ = []\n"),
    ("        if self.train_conf:\n            self.prepare_segformer()\n            self.conf_loss_fn",
     "        if self.train_conf:\n            # CHANGED: no SegFormer sky model (indoor renders have no sky; the SegFormer checkpoint is not on this machine)\n            self.conf_loss_fn"),
    ("            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)\n"
     "            sky_mask[valid_masks] = False\n"
     "            if sky_mask.sum() == 0:\n"
     "                sky_mask_loss = 0.0 * aligned_local_pts.mean()\n"
     "            else:\n"
     "                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))\n",
     "            sky_mask_loss = 0.0 * aligned_local_pts.mean()  # CHANGED: no sky term (see __init__)\n"),
]
new = orig
for a, b in edits:
    assert new.count(a) == 1, a[:60]
    new = new.replace(a, b)
open("ext/vendor_pi3/loss.py", "w", encoding="utf-8", newline="").write(
    f"# VENDORED from {SRC}, file pi3/models/loss.py, with the edits marked CHANGED (diff in loss.diff)\n" + new)
open("ext/vendor_pi3/__init__.py", "w").write('"""pi3 training losses vendored from the pi3 repo\'s training branch."""\n')
d = difflib.unified_diff(orig.splitlines(True), new.splitlines(True), "training:pi3/models/loss.py", "vendor_pi3/loss.py")
open("ext/vendor_pi3/loss.diff", "w", encoding="utf-8", newline="").write("".join(d))
print(open("ext/vendor_pi3/loss.diff").read())
