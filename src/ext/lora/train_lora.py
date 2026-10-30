"""EXP2 §8.6 training controls: LoRA fine-tuning of pi3 / VGGT on the synthetic RS train split (6a) or on its GS
twins (6b), with the models' own training losses and no motion modelling. One JSON config per run.

Recipe (identical for 6a / 6b and every seed; values live in the config):
  frozen DINO encoder and heads; LoRA (rank r, alpha) on qkv / proj / fc1 / fc2 of the multi-view transformer
  (pi3: 36 decoder blocks; VGGT: 24 frame + 24 global aggregator blocks); AdamW, linear warm-up + cosine decay,
  grad-norm clip, bf16 autocast, activation checkpointing per block; one clip per step with a random subset of its
  frames (temporal order kept); the model's own preprocessing and resolution (infer2 runner .load).
  Validation every `val_every` steps (and at step 0) on the val split through the test protocol (runner.run),
  both conditions (rs, gs); early stopping on mean Sim(3) ATE of `select_cond` with `patience` validations.

outputs: ~/rs3d/weights/lora/<run>/step<N>.pt, best.pt     LoRA weights + meta
         ~/rs3d/results/exp2/training/<run>/                config.json, train_log.csv, val_curve.csv,
                                                            val_clips.csv, run.json
usage: python train_lora.py --config <cfg.json> [--smoke N]
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_lora as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="debug: run N steps, validate on 4 clips, write to *_smoke")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    run = cfg["run"] + ("_smoke" if a.smoke else "")
    res_dir = f"{C.ROOT}/results/exp2/training/{run}"
    w_dir = f"{C.ROOT}/weights/lora/{run}"
    if os.path.exists(res_dir) and not os.path.exists(f"{res_dir}/run.json"):  # an earlier attempt died: keep it
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        os.rename(res_dir, f"{res_dir}_aborted_{stamp}")
        if os.path.exists(w_dir):
            os.rename(w_dir, f"{w_dir}_aborted_{stamp}")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{run}] {msg}", flush=True)

    name, seed = cfg["model"], int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from dense_utils import list_clips
    tv = os.path.expanduser(cfg["train_set"])
    tr_df, tr_fail = list_clips(tv, "train")
    va_df, va_fail = list_clips(tv, "val")
    tr_dirs, va_dirs = list(tr_df.dir), list(va_df.dir)
    if a.smoke:
        va_dirs = va_dirs[:2] + va_dirs[-2:]
    log(f"train clips {len(tr_dirs)} ({tr_fail} failed at generation), val clips {len(va_dirs)} ({va_fail} failed)")

    C.wait_for_vram(cfg["vram_need_gb"] + cfg.get("vram_margin_gb", 12), log)
    runner = C.make_runner(name)
    model = runner.model
    pm = C.pixel_map(runner)
    log(f"pixel map {pm}")
    torch.manual_seed(seed)  # LoRA A init depends on the seed
    C.add_lora(model, name, cfg["rank"], cfg["alpha"], cfg.get("dropout", 0.0))
    params = C.prepare_for_training(name, model)
    n_lora = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"LoRA params {n_lora / 1e6:.2f} M of {n_all / 1e6:.1f} M; {len(params)} tensors")

    meta = dict(model=name, rank=cfg["rank"], alpha=cfg["alpha"], dropout=cfg.get("dropout", 0.0),
                targets=C.LORA_TARGETS[name], base_ckpt=runner.ckpt, run=run, seed=seed, train_cond=cfg["train_cond"])
    full_cfg = dict(cfg, run=run, git_commit=C.git_commit(), torch=torch.__version__, pixel_map=pm,
                    n_train_clips=len(tr_dirs), n_val_clips=len(va_dirs), n_lora_params=n_lora,
                    host_free_vram_gb_at_start=C.free_vram_gb(), smoke=a.smoke)
    with open(f"{res_dir}/config.json", "w") as f:
        json.dump(full_cfg, f, indent=1)

    max_steps = a.smoke or cfg.get("stop_after", cfg["max_steps"])  # stop_after: pilots (same schedule, cut short)
    # the sample sequence is drawn for the full schedule (cfg max_steps) and cut at max_steps, so a pilot / a
    # stopped run sees exactly the first samples of the full run with the same seed
    ds = C.ClipFrames(tr_dirs, cfg["train_cond"], seed, cfg["max_steps"], cfg["frames"], runner.load, pm)
    ds.items = ds.items[:max_steps]
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.get("workers", 4),
                                     pin_memory=True, prefetch_factor=4, persistent_workers=False)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, C.lr_lambda(cfg["warmup"], cfg["max_steps"]))
    conf = bool(cfg["conf_loss"])
    loss_fn = C.pi3_loss_fn(conf).cuda() if name == "pi3" else None
    dev = "cuda"

    vf = open(f"{res_dir}/val_curve.csv", "a", newline="")
    vw = None
    cf = open(f"{res_dir}/val_clips.csv", "a", newline="")
    cw = None
    tf = open(f"{res_dir}/train_log.csv", "a", newline="")
    tw = None
    best = dict(step=None, score=np.inf)
    bad = 0
    history = []

    def do_val(step):
        nonlocal vw, cw, bad
        tv0 = time.time()
        rows = C.validate(runner, va_dirs, cfg["val_conds"], pm)
        for r in rows:
            r["step"] = step
            if cw is None:
                cw = csv.DictWriter(cf, fieldnames=list(r))
                cw.writeheader()
            cw.writerow(r)
        cf.flush()
        rec = dict(step=step, val_s=round(time.time() - tv0, 1))
        for c in cfg["val_conds"]:
            for k, v in C.summarize_val(rows, c).items():
                rec[f"{c}_{k}"] = v
        score = rec[f"{cfg['select_cond']}_{cfg['select_metric']}"]
        path = f"{w_dir}/step{step:05d}.pt"
        C.save_lora(path, model, dict(meta, step=step, val=rec))
        improved = score < best["score"]
        if improved:
            best.update(step=step, score=score, rec=rec)
            shutil.copyfile(path, f"{w_dir}/best.pt")
            bad = 0
        elif step > 0:
            bad += 1
        rec["best_step"] = best["step"]
        rec["bad_count"] = bad
        if vw is None:
            vw = csv.DictWriter(vf, fieldnames=list(rec))
            vw.writeheader()
        vw.writerow(rec)
        vf.flush()
        history.append(rec)
        log("VAL " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        return bad >= cfg["patience"]

    stop_reason = "max_steps"
    do_val(0)
    torch.cuda.reset_peak_memory_stats()
    n_skip_nan = n_skip_big = 0
    step = 0
    t_last = time.time()
    def train_step(b):
        """forward, loss, backward, clipped AdamW step; returns (loss, grad norm, loss terms, skip reason)."""
        if name == "pi3":
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = C.pi3_forward(model, b["imgs"], conf)
            loss, det = C.pi3_loss(loss_fn, pred, b)  # pi3 trainer: loss outside autocast
        else:
            bb = C.vggt_batch(b, dev)  # normalisation on CPU, as VGGT's trainer
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # VGGT trainer: forward + loss in autocast
                pred = C.vggt_forward(model, bb["images"])
                loss, det = C.vggt_loss(pred, bb, conf)
        lv = float(loss)
        gn = float("nan")
        if not np.isfinite(lv):
            return lv, gn, det, "nonfinite"
        if name == "pi3" and lv > cfg.get("pi3_clip_loss", 10.0):  # pi3 trainer zeroes losses > clip_loss (10)
            return lv, gn, det, "clip_loss"
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]))
        if not np.isfinite(gn):
            return lv, gn, det, "nonfinite_grad"
        opt.step()
        return lv, gn, det, ""

    n_oom = 0
    for b in dl:
        step += 1
        for attempt in range(30):  # shared GPU: survive brief memory dips by waiting and retrying the same batch
            try:
                lv, gn, det, skipped = train_step(b)
                break
            except torch.cuda.OutOfMemoryError:
                n_oom += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log(f"CUDA OOM at step {step} (attempt {attempt + 1}); free VRAM {C.free_vram_gb():.1f} GB; retry in 60 s")
                time.sleep(60)
        else:
            raise RuntimeError(f"step {step}: 30 OOM retries")
        if skipped.startswith("nonfinite"):
            n_skip_nan += 1
        elif skipped == "clip_loss":
            n_skip_big += 1
        opt.zero_grad(set_to_none=True)
        sched.step()
        dt = time.time() - t_last
        t_last = time.time()
        rec = dict(step=step, loss=lv, grad_norm=gn, lr=sched.get_last_lr()[0], n_frames=int(b["imgs"].shape[1]),
                   clip=b["clip"][0], skipped=skipped, step_s=round(dt, 3),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), **det)
        if tw is None:
            tw = csv.DictWriter(tf, fieldnames=list(rec), extrasaction="ignore")
            tw.writeheader()
        tw.writerow(rec)
        if step % 10 == 0:
            tf.flush()
        if step % 25 == 0 or step <= 3:
            log(f"step {step} loss {lv:.4f} gn {gn:.3f} lr {rec['lr']:.2e} n {rec['n_frames']} {dt:.2f}s "
                f"peak {rec['peak_gb']} GB skips nan/big {n_skip_nan}/{n_skip_big} "
                + " ".join(f"{k}={v:.4f}" for k, v in det.items()))
        if n_skip_nan > 50:
            stop_reason = "too_many_nonfinite"
            break
        val_every = max(1, a.smoke // 2) if a.smoke else cfg["val_every"]
        if step % val_every == 0 or step == max_steps:
            if do_val(step):
                stop_reason = f"early_stop (no improvement in {cfg['patience']} validations)"
                break
    info = dict(run=run, model=name, train_cond=cfg["train_cond"], seed=seed, conf_loss=conf, stop_reason=stop_reason,
                last_step=step, best_step=best["step"], best_score=best["score"],
                best_val=best.get("rec"), n_skip_nonfinite=n_skip_nan, n_skip_clip_loss=n_skip_big, n_oom_retries=n_oom,
                peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30, wall_h=(time.time() - t0) / 3600,
                best_ckpt=f"{w_dir}/best.pt", git_commit=full_cfg["git_commit"])
    with open(f"{res_dir}/run.json", "w") as f:
        json.dump(info, f, indent=1, default=float)
    log("DONE " + json.dumps(info, default=float))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
