"""EXP2 2H: summary tables and the pre-registered decision (brief §10; DEVIATIONS P2-P6, D11-D13).

Reads the per-window / per-clip CSVs written by the EXP2 runs (whatever exists; missing parts are listed, never
filled in) and writes, under results/exp2/:
  gap/<ds>_bins.csv          GS vs RS median ATE by speed bin: n, medians with bootstrap CIs, paired ratio CI, Wilcoxon
  gap/slopes.csv             log(ATE) vs omega slopes with a cluster bootstrap (no-twin datasets; ETH3D as a GS reference)
  oracle/<ds>_ladder.csv     V1 (gs), V2 (rs), V3, V4 (+V5 dense on synthetic) per bin, with the share of the gap closed
  oracle/<ds>_sensitivity.csv  oracle with readout x{0.5,0.75,1.25,1.5} and the wrong scan direction
  oracle/synth_dense.csv     dense metrics per condition on the synthetic core test (V5 included)
  baselines/<ds>_recovery.csv  share of the oracle gain each existing method recovers, with paired bootstrap CIs
  stress/summary.csv         every stress set: median ATE / chamfer per model and condition, RS/GS ratios
  decision.json, decision.md Q1-Q5 with their evidence and the GO / PIVOT / STOP outcome
Condition names are unified to the synthetic spelling (gs, rs, v3, v4_r0.75, v4_dir, se2p-pi3, u2d-..., ...).

usage: python analyze_exp2.py [--root ~/rs3d/results/exp2]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stats2 import boot_median_ci, boot_ratio_ci, holm, paired_wilcoxon, recovery_ci, slope_cluster_boot

BINS = ["slow", "medium", "fast"]
MAIN_MODELS = ["pi3", "vggt"]
SENS = ["v4_r0.5", "v4_r0.75", "v4_r1.25", "v4_r1.5", "v4_dir", "v3_r0.5", "v3_r0.75", "v3_r1.25", "v3_r1.5", "v3_dir"]


def unify(c):
    c = str(c)
    m = re.fullmatch(r"(SE2PT?)_(\w+?)(_shuf)?", c)
    if m:
        return f"{m.group(1).lower()}-{m.group(2)}" + ("-shuf" if m.group(3) else "")
    m = re.fullmatch(r"(V[34])(?:_r([0-9.]+))?(?:_(bt))?", c)
    if m:
        return m.group(1).lower() + (f"_r{m.group(2)}" if m.group(2) else "") + ("_dir" if m.group(3) else "")
    return {"GS": "gs", "RS": "rs", "RSshuf": "rsshuf"}.get(c, c.lower() if c.startswith("U2D") else c)


def load(path, **fix):
    if not os.path.exists(path):
        MISSING.append(os.path.relpath(path, ROOT))
        return None
    df = pd.read_csv(path)
    if "camera" in df and "cond" not in df:
        df = df.rename(columns={"camera": "cond"})
    df["cond"] = df["cond"].map(unify)
    for k, v in fix.items():
        df[k] = v
    return df


def pivot(df, model, conds, metric="ATE"):
    d = df[(df.model == model) & df.cond.isin(conds)].pivot_table(index="key", columns="cond", values=metric)
    return d.dropna(subset=[c for c in conds if c in d], how="any") if all(c in d for c in conds) else None


def bins_table(df, ds, metric="ATE"):
    rows = []
    for m in sorted(df.model.unique()):
        P = pivot(df, m, ["gs", "rs"], metric)
        if P is None:
            continue
        bmap = df[df.model == m].drop_duplicates("key").set_index("key")["bin"]
        for b in BINS + ["all"]:
            Q = P if b == "all" else P[bmap.reindex(P.index).values == b]
            if len(Q) < 5:
                continue
            mg, lg, hg = boot_median_ci(Q.gs)
            mr, lr, hr = boot_median_ci(Q.rs)
            r, rl, rh = boot_ratio_ci(Q.rs.values, Q.gs.values)
            rows.append(dict(dataset=ds, model=m, metric=metric, bin=b, n=len(Q), median_gs=mg, gs_lo=lg, gs_hi=hg,
                             median_rs=mr, rs_lo=lr, rs_hi=hr, ratio=r, ratio_lo=rl, ratio_hi=rh,
                             wilcoxon_p=paired_wilcoxon(Q.rs, Q.gs), frac_rs_worse=float((Q.rs > Q.gs).mean())))
    return pd.DataFrame(rows)


def ladder(df, ds, conds):
    rows = []
    for m in sorted(df.model.unique()):
        have = [c for c in conds if c in set(df[df.model == m].cond)]
        if "rs" not in have or "gs" not in have:
            continue
        P = pivot(df, m, have)
        if P is None:
            continue
        bmap = df[df.model == m].drop_duplicates("key").set_index("key")["bin"]
        for b in BINS + ["all"]:
            Q = P if b == "all" else P[bmap.reindex(P.index).values == b]
            if len(Q) < 5:
                continue
            gap = Q.rs.median() - Q.gs.median()
            for c in have:
                med, lo, hi = boot_median_ci(Q[c])
                rows.append(dict(dataset=ds, model=m, bin=b, cond=c, n=len(Q), median_ATE=med, lo=lo, hi=hi,
                                 gap_closed=(Q.rs.median() - med) / gap if gap > 0 else np.nan,
                                 wilcoxon_vs_rs=paired_wilcoxon(Q[c], Q.rs) if c != "rs" else np.nan))
    return pd.DataFrame(rows)


def q1(tum, ctrl, syn, real_slopes):
    rows = []
    for ds, df, factor_src in (("tum", tum, ctrl), ("synthetic", syn, None)):
        if df is None:
            continue
        fast = df[df.bin == "fast"]
        for m in sorted(df.model.unique()):
            P = pivot(fast, m, ["gs", "rs"])
            if P is None or len(P) < 5:
                continue
            r, lo, hi = boot_ratio_ci(P.rs.values, P.gs.values)
            f = 1.0
            if factor_src is not None:
                C = pivot(factor_src, m, ["gs", "rs"])
                f = C.rs.median() / C.gs.median() if C is not None and len(C) >= 5 else np.nan
            rows.append(dict(dataset=ds, model=m, criterion="fast-bin RS/GS ratio / control ratio", n=len(P),
                             ratio=r, ratio_lo=lo, ratio_hi=hi, control_factor=f, value=r / f,
                             counts=bool(r / f >= 1.3) if np.isfinite(f) else False, real=ds != "synthetic"))
    sl = real_slopes[real_slopes.role == "RS"] if len(real_slopes) else real_slopes
    if len(sl):
        sl = sl.copy()
        sl["p_holm"] = holm(sl.p_one_sided.values)  # Holm family: every (no-twin dataset x model) slope test (P3)
        for r in sl.itertuples():
            rows.append(dict(dataset=r.dataset, model=r.model, criterion="log(ATE)~omega slope, Holm p", n=r.n,
                             value=r.slope, ratio_lo=r.slope_lo, ratio_hi=r.slope_hi, p_holm=r.p_holm,
                             counts=bool(r.p_holm < 0.05 and r.slope > 0), real=True))
    t = pd.DataFrame(rows)
    if t.empty:
        return t, dict(answer="UNKNOWN", reason="no gap results yet")
    per_ds = t[t.counts].groupby("dataset").model.nunique()
    real_ok = [d for d in per_ds.index if d != "synthetic" and per_ds[d] >= 2]
    syn_ok = per_ds.get("synthetic", 0) >= 2
    ans = "YES" if len(real_ok) >= 2 and syn_ok else "NO"
    return t, dict(answer=ans, real_datasets_with_2_models=real_ok, synthetic_models_counting=int(per_ds.get("synthetic", 0)),
                   rule="RS degrades (adjusted fast ratio >= 1.3 or Holm slope p < 0.05) on >= 2 real datasets + "
                        "synthetic, for >= 2 models each")


def slopes(dfs):
    rows = []
    for ds, df, cond, role in dfs:
        if df is None:
            continue
        for m in sorted(df.model.unique()):
            d = df[(df.model == m) & (df.cond == cond)].dropna(subset=["ATE", "omega_mean"])
            d = d[d.ATE > 0]
            if len(d) < 20:
                continue
            s, lo, hi, p = slope_cluster_boot(d.omega_mean.values, d.ATE.values, d.seq.astype(str).values)
            rows.append(dict(dataset=ds, model=m, cond=cond, role=role, n=len(d), n_seqs=d.seq.nunique(), slope=s,
                             slope_lo=lo, slope_hi=hi, p_one_sided=p))
    return pd.DataFrame(rows)


def recovery_table(df, ds, extra=None):
    """Q2 per (dataset, model): oracle gain significance and every method's recovery."""
    rows, pairs = [], []
    for m in sorted(df.model.unique()):
        P = pivot(df, m, ["rs", "v4"])
        if P is None or len(P) < 10:
            continue
        pairs.append(dict(dataset=ds, model=m, n=len(P), med_v2=P.rs.median(), med_v4=P.v4.median(),
                          gain=P.rs.median() - P.v4.median(), p=paired_wilcoxon(P.rs, P.v4)))
        sub = df[df.model == m]
        meths = sorted(c for c in sub.cond.unique()
                       if c.startswith(("u2d", "se2p", "lora6a", "lora6b")) or c in ("colmap", "glomap", "rsshuf"))
        if extra is not None:
            meths += sorted(extra.cond.unique())
        for c in meths:
            src = sub if c in set(sub.cond) else extra
            Q = pd.concat([P, src[src.cond == c].drop_duplicates("key").set_index("key").ATE.rename(c)], axis=1).dropna()
            if len(Q) < 10:
                continue
            r, lo, hi = recovery_ci(Q.rs.values, Q[c].values, Q.v4.values)
            rows.append(dict(dataset=ds, model=m, method=c, n=len(Q), median_ATE=Q[c].median(), recovery=r,
                             recovery_lo=lo, recovery_hi=hi, ordered=not c.endswith("shuf"),
                             family="lora6a" if c.startswith("lora6a") else c))
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def q2(rec, pairs):
    if pairs.empty:
        return pd.DataFrame(), dict(answer="UNKNOWN", reason="no oracle results yet")
    pairs = pairs.copy()
    pairs["p_holm"] = holm(pairs.p.values)
    pairs["eligible"] = (pairs.p_holm < 0.05) & (pairs.gain > 0)
    out = []
    for r in pairs.itertuples():
        if not r.eligible:
            out.append(dict(dataset=r.dataset, model=r.model, eligible=False, reason="oracle gain not significant"))
            continue
        R = rec if rec.empty else rec[(rec.dataset == r.dataset) & (rec.model == r.model) & rec.ordered
                                      & (rec.method != "rsshuf")]
        if R.empty:
            out.append(dict(dataset=r.dataset, model=r.model, eligible=True, best=None, best_recovery=np.nan))
            continue
        fam = R.groupby("family").recovery.mean()  # LoRA 6a: mean over its 3 seeds (D13)
        best = fam.idxmax()
        out.append(dict(dataset=r.dataset, model=r.model, eligible=True, best=best, best_recovery=float(fam.max()),
                        methods_compared=int(len(fam))))
    t = pd.DataFrame(out)
    el = t[t.eligible & t.best_recovery.notna()] if "best_recovery" in t else t.iloc[0:0]
    if el.empty:
        return t, dict(answer="UNKNOWN", reason="no eligible pair with baseline results yet")
    share = float((el.best_recovery < 0.6).mean())
    return t, dict(answer="YES" if share >= 0.5 else "NO", share_pairs_best_below_60pct=share, n_pairs=int(len(el)),
                   rule="best existing method recovers < 60% of the oracle gain on >= half of the eligible pairs",
                   note="baseline coverage is listed per pair in baselines/*_recovery.csv")


def q3(syn):
    if syn is None or "chamfer" not in syn:
        return pd.DataFrame(), dict(answer="UNKNOWN", reason="no synthetic dense results yet")
    rows = []
    for m in sorted(syn.model.unique()):
        P = pivot(syn, m, ["v3", "v4", "v5"], "chamfer")
        if P is None or len(P) < 10:
            continue
        better = "v3" if P.v3.median() <= P.v4.median() else "v4"
        rel = 1 - P.v5.median() / P[better].median()
        rows.append(dict(model=m, n=len(P), med_v3=P.v3.median(), med_v4=P.v4.median(), med_v5=P.v5.median(),
                         better_of_v3v4=better, v5_reduction=rel,
                         p=paired_wilcoxon(P.v5, P[better], alternative="less")))
    t = pd.DataFrame(rows)
    if t.empty:
        return t, dict(answer="UNKNOWN", reason="no V5 results yet")
    t["p_holm"] = holm(t.p.values)
    t["v5_wins"] = (t.v5_reduction >= 0.10) & (t.p_holm < 0.05)
    main = t[t.model.isin(MAIN_MODELS)]
    k = int(main.v5_wins.sum())
    ans = "YES" if k == len(MAIN_MODELS) else ("MIXED" if k else "NO")
    return t, dict(answer=ans, rule="V5 median chamfer >= 10% below the better of V3/V4, Holm p < 0.05, for both "
                                    "pi3 and VGGT (D13)")


def q5(df_by_ds, pairs):
    rows = []
    el = pairs[pairs.eligible] if "eligible" in pairs else pairs.iloc[0:0]
    for r in el.itertuples():
        df = df_by_ds[r.dataset]
        for base in ("v4", "v3"):
            conds = ["rs", base, f"{base}_r0.75", f"{base}_r1.25"]
            P = pivot(df, r.model, conds)
            if P is None or len(P) < 10:
                continue
            gain = P.rs.median() - P[base].median()
            row = dict(dataset=r.dataset, model=r.model, oracle=base, n=len(P), gain=gain)
            for f in ("r0.5", "r0.75", "r1.25", "r1.5", "dir"):
                c = f"{base}_{f}"
                Q = pivot(df, r.model, ["rs", base, c])
                if Q is not None and len(Q) >= 10:
                    row[f"lost_{f}"] = (Q[c].median() - Q[base].median()) / (Q.rs.median() - Q[base].median())
            row["lost_pm25"] = max(row.get("lost_r0.75", np.nan), row.get("lost_r1.25", np.nan))
            rows.append(row)
            break  # P6: V4 where possible, V3 only if V4 is not
    t = pd.DataFrame(rows)
    if t.empty:
        return t, dict(answer="UNKNOWN", reason="no sensitivity results yet")
    share = float((t.lost_pm25 > 0.20).mean())
    return t, dict(answer="YES" if share >= 0.5 else "NO", share_pairs_lost_over_20pct=share, n_pairs=len(t),
                   rule="+-25% readout error costs > 20% of the oracle gain on >= half of the eligible pairs")


def vs_base(df, ds, base, metric="ATE"):
    """Every condition against a baseline condition on the same windows: median, paired ratio CI, Wilcoxon."""
    rows = []
    if df is None:
        return pd.DataFrame()
    for m in sorted(df.model.unique()):
        sub = df[df.model == m]
        for c in sorted(set(sub.cond) - {base}):
            P = pivot(sub, m, [base, c], metric)
            if P is None or len(P) < 10:
                continue
            r, lo, hi = boot_ratio_ci(P[c].values, P[base].values)
            rows.append(dict(dataset=ds, model=m, base=base, cond=c, n=len(P), median_base=P[base].median(),
                             median_cond=P[c].median(), ratio=r, ratio_lo=lo, ratio_hi=hi,
                             wilcoxon_p=paired_wilcoxon(P[c], P[base]), frac_worse=float((P[c] > P[base]).mean())))
    return pd.DataFrame(rows)


def stress_summary():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/stress/*/per_clip*.csv")):
        st = os.path.basename(os.path.dirname(f)) + os.path.basename(f)[len("per_clip"):-4]
        df = pd.read_csv(f)
        for (m, c), g in df.groupby(["model", "cond"]):
            rows.append(dict(set=st, model=m, cond=c, n=len(g), median_ATE=g.ATE.median(),
                             median_AUC30=g.AUC30.median() if "AUC30" in g else np.nan,
                             median_chamfer=g.chamfer.median() if "chamfer" in g else np.nan,
                             median_plan_rms=g.plan_rms.median() if "plan_rms" in g else np.nan))
    t = pd.DataFrame(rows)
    if not t.empty:
        gs = t[t.cond == "gs"].set_index(["set", "model"]).median_ATE
        t["ratio_vs_gs"] = [r.median_ATE / gs.get((r.set, r.model), np.nan) for r in t.itertuples()]
    return t


def main():
    global ROOT, MISSING
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/rs3d/results/exp2"))
    a = ap.parse_args()
    ROOT, MISSING = a.root, []
    for d in ("gap", "oracle", "baselines", "stress"):
        os.makedirs(f"{ROOT}/{d}", exist_ok=True)

    tum = load(f"{ROOT}/gap/tum/per_window_all.csv")
    for f in ("per_window_newmodels.csv",):
        x = load(f"{ROOT}/gap/tum/{f}")
        tum = x if tum is None else (pd.concat([tum, x]) if x is not None else tum)
    for f in glob.glob(f"{ROOT}/baselines/*/tum_per_window*.csv"):  # helper outputs (U2D, classical, LoRA)
        x = load(f)
        tum = x if tum is None else pd.concat([tum, x])
    ctrl = load(f"{ROOT}/gap/tum/per_window_control.csv")
    advio = load(f"{ROOT}/gap/advio/per_window.csv")
    eth3d = load(f"{ROOT}/gap/eth3d/per_window.csv")
    syn = load(f"{ROOT}/synth/core_test/per_clip.csv")
    for f in glob.glob(f"{ROOT}/baselines/*/synth_core_test*.csv"):
        x = load(f)
        syn = x if syn is None else pd.concat([syn, x])
    if tum is not None:
        tum = tum.drop_duplicates(["model", "cond", "key"], keep="last")
    if syn is not None:
        syn = syn.drop_duplicates(["model", "cond", "key"], keep="last")

    for ds, df in (("tum", tum), ("synthetic", syn), ("advio", advio), ("eth3d", eth3d)):
        if df is not None and {"gs", "rs"} <= set(df.cond):
            bins_table(df, ds).to_csv(f"{ROOT}/gap/{ds}_bins.csv", index=False)
    sl = slopes([("advio", advio, "rs", "RS"), ("eth3d", eth3d, "gs", "GS reference"),
                 ("tum", tum, "gs", "GS reference (not counted)"), ("tum", tum, "rs", "RS twin (not counted)")])
    sl.to_csv(f"{ROOT}/gap/slopes.csv", index=False)
    t1, d1 = q1(tum, ctrl, syn, sl)
    t1.to_csv(f"{ROOT}/gap/q1_table.csv", index=False)

    ladder_conds = ["gs", "rs", "v3", "v4"]  # V5 has no pose metrics; it is in oracle/synth_dense.csv
    for ds, df in (("tum", tum), ("synthetic", syn)):
        if df is not None:
            ladder(df, ds, ladder_conds).to_csv(f"{ROOT}/oracle/{ds}_ladder.csv", index=False)
            ladder(df, ds, ["rs", "gs", "v3", "v4"] + SENS).to_csv(f"{ROOT}/oracle/{ds}_sensitivity.csv", index=False)
    if syn is not None and "chamfer" in syn:
        dense_cols = [c for c in ("chamfer", "acc", "comp", "corr_err", "plan_rms", "absrel", "delta1") if c in syn]
        g = syn.groupby(["model", "cond"])
        dt = g[dense_cols].median().join(g.size().rename("n"))
        dt.reset_index().to_csv(f"{ROOT}/oracle/synth_dense.csv", index=False)

    recs, pairs = [], []
    for ds, df in (("tum", tum), ("synthetic", syn)):
        if df is not None:
            r, p = recovery_table(df, ds)
            r.to_csv(f"{ROOT}/baselines/{ds}_recovery.csv", index=False)
            recs.append(r)
            pairs.append(p)
    rec = pd.concat(recs) if recs else pd.DataFrame()
    pairs = pd.concat(pairs) if pairs else pd.DataFrame()
    t2, d2 = q2(rec, pairs)
    t2.to_csv(f"{ROOT}/baselines/q2_table.csv", index=False)
    if not pairs.empty:
        pairs["p_holm"] = holm(pairs.p.values)
        pairs["eligible"] = (pairs.p_holm < 0.05) & (pairs.gain > 0)
        pairs.to_csv(f"{ROOT}/oracle/oracle_gain_pairs.csv", index=False)
    t3, d3 = q3(syn)
    t3.to_csv(f"{ROOT}/oracle/q3_table.csv", index=False)
    t5, d5 = q5({"tum": tum, "synthetic": syn}, pairs)
    t5.to_csv(f"{ROOT}/oracle/q5_table.csv", index=False)
    st = stress_summary()
    st.to_csv(f"{ROOT}/stress/summary.csv", index=False)

    # no-harm (brief §9 item 11): methods given GS frames vs the plain model on the same GS frames
    nh = [vs_base(load(f"{ROOT}/gap/tum/per_window_noharm.csv"), "tum_cam0", "gs"),
          vs_base(load(f"{ROOT}/gap/eth3d/per_window_conds.csv"), "eth3d", "gs"),
          vs_base(load(f"{ROOT}/synth/core_test/per_clip_noharm.csv"), "synthetic_gs", "gs")]
    for f in glob.glob(f"{ROOT}/baselines/*/noharm*.csv"):  # helper outputs (U2D / LoRA on GS input)
        x = load(f)
        if x is not None and "dataset" in x:
            for ds_name, g in x.groupby("dataset"):
                nh.append(vs_base(g, ds_name, "gs"))
    pd.concat([t for t in nh if not t.empty] or [pd.DataFrame()]).to_csv(f"{ROOT}/stress/noharm.csv", index=False)
    # stress item 3: RS/GS by motion kind and by scene depth (terciles of the clip's median GS depth)
    if syn is not None:
        rows = []
        sd = syn.copy()
        cuts = sd.drop_duplicates("key").depth_median.quantile([1 / 3, 2 / 3]).values
        sd["depth_bin"] = np.where(sd.depth_median < cuts[0], "near", np.where(sd.depth_median < cuts[1], "mid", "far"))
        for col in ("kind", "depth_bin"):
            for m in sorted(sd.model.unique()):
                for g, sub in sd[sd.model == m].groupby(col):
                    P = pivot(sub, m, ["gs", "rs"])
                    if P is None or len(P) < 5:
                        continue
                    r, lo, hi = boot_ratio_ci(P.rs.values, P.gs.values)
                    rows.append(dict(by=col, group=g, model=m, n=len(P), median_gs=P.gs.median(),
                                     median_rs=P.rs.median(), ratio=r, ratio_lo=lo, ratio_hi=hi,
                                     depth_cuts_m=f"{cuts[0]:.2f}/{cuts[1]:.2f}" if col == "depth_bin" else ""))
        pd.DataFrame(rows).to_csv(f"{ROOT}/stress/core_by_kind_depth.csv", index=False)
    # ADVIO: gyroscope straightening over assumed readout / direction, and self-estimate, vs plain RS
    vs_base(load(f"{ROOT}/gap/advio/per_window_conds.csv"), "advio", "rs").to_csv(
        f"{ROOT}/baselines/advio_conditions.csv", index=False)

    ck = open(f"{ROOT}/CHECKPOINT_2A.md", encoding="utf-8").read() if os.path.exists(f"{ROOT}/CHECKPOINT_2A.md") else ""
    m = re.search(r"\b(OPEN|CROWDED|TAKEN)\b[^\n]*?(\d+)\s*%", ck)
    d4 = dict(answer=m.group(1) if m else "UNKNOWN", confidence_pct=int(m.group(2)) if m else None,
              source="CHECKPOINT_2A.md")
    q4_ok = d4["answer"] == "OPEN" or d4["answer"] == "CROWDED"
    if d1["answer"] == "NO":
        outcome = "STOP"
    elif d2["answer"] == "NO" or d4["answer"] == "TAKEN":
        outcome = "PIVOT"
    elif d1["answer"] == "YES" and d2["answer"] == "YES" and q4_ok:
        outcome = "GO"
    else:
        outcome = "UNDECIDED"
    dec = dict(Q1=d1, Q2=d2, Q3=d3, Q4=d4, Q5=d5, outcome=outcome, missing_inputs=MISSING)
    with open(f"{ROOT}/decision.json", "w") as f:
        json.dump(dec, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(dec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    for name, t in (("Q1", t1), ("Q2", t2), ("Q3", t3), ("Q5", t5)):
        print(f"\n== {name}\n" + (t.round(4).to_string() if not t.empty else "(empty)"))


if __name__ == "__main__":
    main()
