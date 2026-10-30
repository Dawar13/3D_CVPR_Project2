"""Step-0 identity check for the LoRA controls (brief §8.6): with LoRA B = 0 the model must reproduce the frozen
model exactly. Checks, on one val clip (16 frames, RS) per model:
  1. base runner.run                      vs  runner.run after LoRA injection (seeded A, B = 0)
  2. training forward (grad mode, checkpointed blocks) vs runner.run outputs (same numbers)
  3. a saved step-0 checkpoint loaded through models_lora into a fresh runner  vs  base runner.run
  4. gradients at step 0: dL/dB != 0, dL/dA == 0 (B = 0), base weights get no grad
writes ~/rs3d/results/exp2/training/identity_check_<model>.json
usage: python check_identity.py --model pi3|vggt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_lora as C  # noqa: E402


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.nanmax(np.abs(a - b))) if a.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pi3", "vggt"])
    ap.add_argument("--clip", default="")
    a = ap.parse_args()
    from dense_utils import list_clips
    va, _ = list_clips(f"{C.ROOT}/data/synth/core_trainval", "val")
    d = a.clip or va.dir.iloc[0]
    paths = [f"{d}/rs/{k:02d}.png" for k in range(16)]
    out = dict(model=a.model, clip=d, git_commit=C.git_commit())

    runner = C.make_runner(a.model)
    pm = C.pixel_map(runner)
    x = runner.load(paths)
    base = runner.run(x, dense=True)

    torch.manual_seed(0)
    C.add_lora(runner.model, a.model, 16, 32)
    params = C.prepare_for_training(a.model, runner.model)
    nB = [p for n, p in runner.model.named_parameters() if "lora_B" in n]
    nA = [p for n, p in runner.model.named_parameters() if "lora_A" in n]
    out["lora_B_all_zero"] = bool(all(float(p.abs().max()) == 0 for p in nB))
    out["lora_A_nonzero"] = bool(all(float(p.abs().max()) > 0 for p in nA))
    out["n_lora_tensors"] = len(params)
    lo = runner.run(x, dense=True)
    out["run_vs_base_maxdiff"] = {k: maxdiff(base[k], lo[k]) for k in base}

    # training forward in grad mode (checkpointed) + loss + backward
    g = C.load_clip_gt(d, "rs", np.arange(16), pm)
    b = dict(imgs=x[None].float(), local=torch.from_numpy(g["local"])[None], depth=torch.from_numpy(g["depth"])[None],
             valid=torch.from_numpy(g["valid"])[None], c2w=torch.from_numpy(g["c2w"]).float()[None],
             K=torch.from_numpy(g["K"]).float()[None])
    if a.model == "pi3":
        bd = {k: v.cuda() for k, v in b.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = C.pi3_forward(runner.model, bd["imgs"], need_conf=True)
        out["trainfwd_vs_base_maxdiff"] = dict(
            poses=maxdiff(pred["camera_poses"][0].detach().float().cpu(), base["poses"]),
            local_pts=maxdiff(pred["local_points"][0].detach().float().cpu(), base["local_pts"]),
            conf=maxdiff(torch.sigmoid(pred["conf"][0, ..., 0].detach().float()).cpu(), base["conf"]))
        loss, det = C.pi3_loss(C.pi3_loss_fn(conf=True).cuda(), pred, bd)
    else:
        bb = C.vggt_batch(b, "cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = C.vggt_forward(runner.model, bb["images"])
            loss, det = C.vggt_loss(pred, bb, conf=True)
        I = C.infer2_module()
        extr, intr = runner.dec(pred["pose_enc"].detach(), bb["images"].shape[-2:])
        E = np.tile(np.eye(4), (16, 1, 1))
        E[:, :3, :] = extr[0].float().cpu().numpy()
        dpt = I.intr_unproject(pred["depth"][0, ..., 0].detach().float(), intr[0].float()).cpu().numpy()
        # heads run in fp32 in VGGT.forward (training) but the depth head runs under bf16 autocast in infer2's
        # VGGTRunner (README usage), so local points differ at bf16 level; poses use the same fp32 camera head
        out["trainfwd_vs_base_maxdiff"] = dict(poses=maxdiff(np.linalg.inv(E), base["poses"]),
                                               local_pts=maxdiff(dpt, base["local_pts"]))
        out["trainfwd_vs_base_local_pts_median_reldiff"] = float(np.median(
            np.abs(dpt[..., 2] - base["local_pts"][..., 2]) / np.maximum(np.abs(base["local_pts"][..., 2]), 1e-6)))
    out["loss_step0"] = float(loss)
    out["loss_terms_step0"] = det
    loss.backward()
    gB = [float(p.grad.abs().max()) if p.grad is not None else 0.0 for p in nB]
    gA = [float(p.grad.abs().max()) if p.grad is not None else 0.0 for p in nA]
    out["grad_B_nonzero_frac"] = float(np.mean(np.array(gB) > 0))
    out["grad_A_max"] = float(max(gA))
    out["base_params_with_grad"] = sum(1 for n, p in runner.model.named_parameters()
                                       if "lora_" not in n and p.grad is not None)
    runner.model.zero_grad(set_to_none=True)
    del pred, loss

    # checkpoint round trip through models_lora
    ck = f"{C.ROOT}/weights/lora/identity_check_{a.model}_step0.pt"
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    C.save_lora(ck, runner.model, dict(model=a.model, rank=16, alpha=32, dropout=0.0, step=0, run="identity_check"))
    del runner, params, nB, nA, x, b
    bd = bb = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["LORA_CKPT"] = ck
    import models_lora
    r2 = models_lora.EXTRA_RUNNERS[f"{a.model}_lora"]()
    assert C.pixel_map(r2) == pm
    lo2 = r2.run(r2.load(paths), dense=True)
    out["models_lora_ckpt_vs_base_maxdiff"] = {k: maxdiff(base[k], lo2[k]) for k in base}
    exact = all(v == 0 for v in out["run_vs_base_maxdiff"].values()) and \
        all(v == 0 for v in out["models_lora_ckpt_vs_base_maxdiff"].values())
    out["PASS_exact_identity"] = bool(exact and out["lora_B_all_zero"] and out["grad_A_max"] == 0
                                      and out["grad_B_nonzero_frac"] > 0.9 and out["base_params_with_grad"] == 0)
    p = f"{C.ROOT}/results/exp2/training/identity_check_{a.model}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("IDENTITY_DONE", flush=True)


if __name__ == "__main__":
    main()
