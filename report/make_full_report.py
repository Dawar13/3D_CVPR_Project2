"""Builds the combined EXP0 + EXP1 team report: report/RS3D_EXP0_EXP1_Full_Report.pdf

Tables are filled from the result CSVs, so every number matches results/exp0 and results/exp1 (measured).
Run make_figures.py and make_figures_full.py first.
"""
import os
import re

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, SimpleDocTemplate, Spacer

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

from make_pdf import (BLUE, BOX_BLUE, BOX_GRAY, BOX_ORANGE, FRAME_W, INK2, MARGIN, ORANGE, PAGE_W, S, P, box,
                      bullets, colors, fig, numbered, table)

HERE = os.path.dirname(os.path.abspath(__file__))
RES0 = os.path.join(HERE, "..", "results", "exp0")
RES1 = os.path.join(HERE, "..", "results", "exp1")
OUT = os.path.join(HERE, "RS3D_EXP0_EXP1_Full_Report.pdf")

GREEN = colors.HexColor("#1a7f37")
BOX_GREEN = colors.HexColor("#e8f4ee")
MODELS = ["mapanything", "vggt", "pi3"]
NAME = dict(mapanything="MapAnything", vggt="VGGT", pi3="π³")
BINS = ["slow", "medium", "fast"]
DEG = 180 / np.pi
W = FRAME_W

bs0 = pd.read_csv(f"{RES0}/bins_summary.csv")
cs0 = pd.read_csv(f"{RES0}/control_summary.csv")
fine0 = pd.read_csv(f"{RES0}/ate_fine_bins.csv")
reg0 = pd.read_csv(f"{RES0}/regression_logratio.csv")
trend0 = pd.read_csv(f"{RES0}/trend.csv")
s1 = pd.read_csv(f"{RES1}/e1_summary.csv")
pw1 = pd.read_csv(f"{RES1}/per_window_e1.csv")


CELL_S = ParagraphStyle("cell_s", parent=S["cell"], fontSize=8.3, leading=10.5)
CELLB_S = ParagraphStyle("cellb_s", parent=S["cellb"], fontSize=8.3, leading=10.5)


def table_s(rows, widths):
    """Same look as table(), smaller type, for the wide data tables."""
    return table([[Paragraph(c, CELLB_S if i == 0 else CELL_S) if isinstance(c, str) else c for c in r]
                  for i, r in enumerate(rows)], widths)


def pval(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.2f}"


def e0(model, b, metric):
    return bs0[(bs0.model == model) & (bs0.bin == b) & (bs0.metric == metric)].iloc[0]


def e1(model, b, metric, var):
    return s1[(s1.model == model) & (s1.bin == b) & (s1.metric == metric) & (s1.variant == var)].iloc[0]


def ext_fig(path, caption, width=W):
    w, h = PILImage.open(path).size
    return KeepTogether([Image(path, width=width, height=width * h / w), P(caption, "cap")])


def flat(items):
    """Unpack nested KeepTogether blocks: a nested one reports a huge height and forces a page break."""
    out = []
    for f in items:
        out += flat(f._content) if isinstance(f, KeepTogether) else [f]
    return out


def h2_with(title, *flowables):
    """Keep a heading on the same page as what follows it."""
    return KeepTogether(flat([P(title, "h2"), *flowables]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8.5)
    canvas.setFillColor(INK2)
    canvas.drawString(MARGIN, 10 * mm, "RS-3D  ·  EXP0 + EXP1 full results  ·  experiments run 27 Oct 2026  ·  "
                                       "report 29 Oct 2026")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------- tables built from the CSVs
def t_e0_metric(metric, unit):
    scale, fmt = (100, "{:.2f} cm") if unit == "cm" else (1, "{:.2f}°")
    rows = [["Model", "Speed", "Clips", "Normal camera", "Rolling shutter", "Times worse [95% range]",
             "Clips worse on RS", "p"]]
    for m in MODELS:
        for b in BINS:
            r = e0(m, b, metric)
            rows.append([NAME[m] if b == "slow" else "", b, f"{r.n}", fmt.format(r.median_GS * scale),
                         fmt.format(r.median_RS * scale),
                         f"<b>{r.ratio:.2f}×</b> [{r.ratio_ci_lo:.2f}, {r.ratio_ci_hi:.2f}]",
                         f"{r.frac_RS_worse * 100:.0f}%", pval(r.wilcoxon_p)])
    return table_s(rows, [0.14 * W, 0.11 * W, 0.07 * W, 0.13 * W, 0.13 * W, 0.21 * W, 0.11 * W, 0.10 * W])


def omega_range(s):
    lo, hi = map(float, re.findall(r"[\d.]+", s))
    return f"{lo * DEG:.0f}–{hi * DEG:.0f}°/s"


def t_e0_fine():
    rows = [["Model", "Turning speed", "Clips", "Normal camera", "Rolling shutter", "Extra error", "Times worse"]]
    for m in MODELS:
        for i, r in enumerate(fine0[fine0.model == m].itertuples()):
            rows.append([NAME[m] if i == 0 else "", omega_range(r.omega), f"{r.n}", f"{r.GS_cm:.2f} cm",
                         f"{r.RS_cm:.2f} cm", f"+{r.RS_minus_GS_cm:.2f} cm", f"{r.ratio:.2f}×"])
    return table_s(rows, [0.15 * W, 0.17 * W, 0.08 * W, 0.15 * W, 0.15 * W, 0.15 * W, 0.15 * W])


def t_e0_control():
    rows = [["Model", "Speed third", "Clips", "Camera 1", "Camera 2", "Camera 2 ÷ camera 1 [95% range]", "p"]]
    for m in MODELS:
        for i, b in enumerate(["slow", "medium", "fast", "all"]):
            r = cs0[(cs0.model == m) & (cs0.metric == "ATE") & (cs0.bin == b)].iloc[0]
            rows.append([NAME[m] if i == 0 else "", "all clips" if b == "all" else b, f"{r.n}",
                         f"{r.median_cam0 * 100:.2f} cm", f"{r.median_cam1 * 100:.2f} cm",
                         f"<b>{r.ratio:.2f}×</b> [{r.ratio_ci_lo:.2f}, {r.ratio_ci_hi:.2f}]", pval(r.wilcoxon_p)])
    return table_s(rows, [0.15 * W, 0.13 * W, 0.08 * W, 0.14 * W, 0.14 * W, 0.26 * W, 0.10 * W])


def gap(r):
    """Share of the gap closed, or a dash when there is no real gap to close (RS within 5% of GS)."""
    if r.median_RS / r.median_GS < 1.05:
        return "— (no gap)"
    return f"<b>{r.recovered * 100:.0f}%</b> [{r.recovered_ci_lo * 100:.0f}, {r.recovered_ci_hi * 100:.0f}]"


def t_e1_metric(metric, unit, bins=BINS + ["all"]):
    scale, fmt = (100, "{:.2f}") if unit == "cm" else (1, "{:.2f}")
    rows = [["Model", "Speed", f"Normal ({unit})", f"Bent ({unit})", f"Fixed for turning ({unit})",
             "Share removed [95%]", f"Fixed for turning + walking ({unit})", "Share removed [95%]"]]
    for m in MODELS:
        for i, b in enumerate(bins):
            a, c = e1(m, b, metric, "RSU"), e1(m, b, metric, "RSUT")
            rows.append([NAME[m] if i == 0 else "", "all clips" if b == "all" else b,
                         fmt.format(a.median_GS * scale), fmt.format(a.median_RS * scale),
                         fmt.format(a.median_unrolled * scale), gap(a), fmt.format(c.median_unrolled * scale),
                         gap(c)])
    return table_s(rows, [0.13 * W, 0.10 * W, 0.10 * W, 0.09 * W, 0.13 * W, 0.15 * W, 0.15 * W, 0.15 * W])


def t_e1_consistency():
    rows = [["Model", "Speed", "Clips", "Improved: turning", "Improved: turning + walking", "p: turning",
             "p: turning + walking"]]
    for m in MODELS:
        for i, b in enumerate(BINS + ["all"]):
            a, c = e1(m, b, "ATE", "RSU"), e1(m, b, "ATE", "RSUT")
            rows.append([NAME[m] if i == 0 else "", "all clips" if b == "all" else b, f"{a.n}",
                         f"{a.frac_windows_improved * 100:.0f}%", f"{c.frac_windows_improved * 100:.0f}%",
                         pval(a.wilcoxon_p_unrolled_vs_RS), pval(c.wilcoxon_p_unrolled_vs_RS)])
    return table_s(rows, [0.15 * W, 0.12 * W, 0.08 * W, 0.17 * W, 0.2 * W, 0.13 * W, 0.15 * W])


def t_e1_floor():
    rows = [["Model", "Speed", "Turning: plain", "Turning: fairness-adjusted",
             "Turning + walking: plain", "Turning + walking: fairness-adjusted"]]
    for m in MODELS:
        for i, b in enumerate(BINS + ["all"]):
            a, c = e1(m, b, "ATE", "RSU"), e1(m, b, "ATE", "RSUT")
            rows.append([NAME[m] if i == 0 else "", "all clips" if b == "all" else b, f"{a.recovered * 100:.0f}%",
                         f"{a.recovered_vs_control_floor * 100:.0f}%", f"{c.recovered * 100:.0f}%",
                         f"{c.recovered_vs_control_floor * 100:.0f}%"])
    return table_s(rows, [0.15 * W, 0.13 * W, 0.15 * W, 0.2 * W, 0.17 * W, 0.2 * W])


def t_e0_rta():
    rows = [["Model", "Speed", "Normal camera", "Rolling shutter", "Rolling ÷ normal"]]
    for m in MODELS:
        for b in BINS:
            r = e0(m, b, "RTA5")
            rows.append([NAME[m] if b == "slow" else "", b, f"{r.median_GS * 100:.0f}%", f"{r.median_RS * 100:.0f}%",
                         f"{r.ratio:.2f}×"])
    return table(rows, [0.2 * W, 0.15 * W, 0.22 * W, 0.22 * W, 0.21 * W])


def t_reg():
    rows = [["Model", "Score", "Effect of turning speed (per rad/s)", "Effect of walking speed (per m/s)",
             "Times worse with no motion (fitted)"]]
    for r in reg0.itertuples():
        score = "path error" if r.metric == "ATE" else "turning error"
        rows.append([NAME[r.model], score, f"{r.b_omega:+.3f} ± {r.se_omega:.3f}", f"{r.b_v:+.3f} ± {r.se_v:.3f}",
                     f"{r.ratio_at_zero_motion:.2f}×"])
    return table(rows, [0.15 * W, 0.15 * W, 0.24 * W, 0.24 * W, 0.22 * W])


def t_trend():
    rows = [["Model", "Score", "Trend with turning speed (Spearman ρ)", "p"]]
    score = dict(ATE="path error", RPE_rot="turning error", RPE_trans="step error")
    for r in trend0.itertuples():
        rows.append([NAME[r.model], score[r.metric], f"{r.spearman_rho_omega_vs_logratio:+.2f}", pval(r.p)])
    return table(rows, [0.2 * W, 0.2 * W, 0.4 * W, 0.2 * W])


def t_fastest():
    d = pw1[pw1.key == "s9_w0077"].pivot_table(index="model", columns="camera", values="ATE") * 100
    rows = [["Model", "Normal", "Bent", "Straightened: turning", "Straightened: turning + walking"]]
    for m in MODELS:
        rows.append([NAME[m]] + [f"{d.loc[m, c]:.1f} cm" for c in ["GS", "RS", "RSU", "RSUT"]])
    return table(rows, [0.2 * W, 0.15 * W, 0.15 * W, 0.22 * W, 0.28 * W])


# ---------------------------------------------------------------- the document
def build():
    st = []
    fast = {m: e0(m, "fast", "ATE") for m in MODELS}
    g_rsu = {m: e1(m, "fast", "ATE", "RSU") for m in MODELS}
    g_rsut = {m: e1(m, "fast", "ATE", "RSUT") for m in MODELS}
    pf = fine0[fine0.model == "pi3"]

    # ============================================================ cover
    st += [P("RS-3D: Experiments 0 and 1, full results", "title"),
           P("What we tested, every number we measured, and what comes next &nbsp;·&nbsp; for Dawar, Shubh and "
             "Sujal &nbsp;·&nbsp; 29 October 2026", "sub")]
    st.append(box([
        P("The short version", "boxh"),
        *bullets([
            f"<b>The problem is real (EXP0).</b> 3D AI models make bigger mistakes on rolling-shutter photos. "
            f"In fast motion, π³ is <b>{fast['pi3'].ratio:.1f}× worse</b>, VGGT {fast['vggt'].ratio:.1f}× and "
            f"MapAnything {fast['mapanything'].ratio:.1f}×.",
            f"<b>The problem can be fixed (EXP1).</b> Straightening the bent photos with the known camera motion "
            f"removes <b>{min(r.recovered for r in g_rsu.values()) * 100:.0f}–"
            f"{max(r.recovered for r in g_rsu.values()) * 100:.0f}%</b> of the extra error in fast motion "
            f"(turning only), and <b>90% or more</b> when walking is corrected too. Result: <b>GO</b> for all "
            f"three models.",
            "<b>The catch.</b> EXP1 used the rig's gyroscope and the true camera path. A real system often has "
            "neither, and fixing photos with a gyroscope is very likely already published. Our paper must "
            "fix the bending <b>from the images alone</b>.",
            "<b>Next: EXP2.</b> Can a model work out the motion from its own first attempt and fix itself? "
            "The answer decides which paper we write.",
        ]),
    ]))
    st.append(Spacer(1, 8))
    st.append(P("The two experiments at a glance", "h2"))
    st.append(table([
        ["", "EXP0: gap test", "EXP1: fix with known motion"],
        ["Question", "Are models worse on rolling-shutter photos, and worse when moving faster?",
         "If we straighten the photos using the true motion, does the extra error go away?"],
        ["Answer", "Yes for VGGT and π³ (1.5–2.6× in fast motion). MapAnything barely (1.16×).",
         "Yes. 82–99% removed with turning only, 90% or more with turning and walking."],
        ["Pre-agreed verdict", "<b>GRAY</b>, because slow clips were also worse. Explained by EXP1.",
         "<font color='#1a7f37'><b>GO</b></font> for all three models"],
        ["Run time", "about 1 h 50 min", "about 2 h 02 min"],
    ], [0.18 * W, 0.41 * W, 0.41 * W]))
    st.append(Spacer(1, 6))
    st.append(P("All numbers in this report were <b>measured</b>. The few planning estimates are marked "
                "<i>projected</i>. Tables are generated directly from the result files in "
                "<font face='Courier'>results/exp0</font> and <font face='Courier'>results/exp1</font>.", "small"))
    st.append(P("Contents", "h2"))
    st += numbered(["The problem, in one picture", "What we used: data, models, scores",
                    "EXP0: is there a problem?", "EXP1: can it be fixed if we know the motion?",
                    "Where we stand", "The road ahead, and why",
                    "Appendices: glossary, run facts, changes from the plan, extra statistics, technical charts, "
                    "files"])

    # ============================================================ 1. problem
    st.append(PageBreak())
    st.append(P("1. The problem, in one picture", "h1"))
    st.append(P("Most cameras in phones, drones and robots do not take the whole photo at once. They read it "
                "<b>row by row, top to bottom</b>, over about 30 milliseconds. This is a <b>rolling shutter</b>. "
                "If the camera moves during those 30 ms, each row is taken from a slightly different position, so "
                "straight things come out slanted or wobbly."))
    st.append(P("3D AI models such as VGGT, π³ and MapAnything assume every photo was taken in a single instant. "
                "With bent photos they misjudge where the camera was and how the scene is shaped."))
    st.append(fig("f1_sketch.png", "Figure 1. A straight pole photographed while the camera turns. Left: all rows "
                                   "at once. Right: one row at a time."))
    st.append(fig("f2_photos.png", "Figure 2. Real photos from our dataset, taken at the same instant while the "
                                   "camera turned at about 200° per second. Part of the difference is because the "
                                   "two cameras sit 11 cm apart."))
    st.append(box([P("<b>Our project idea:</b> teach the 3D model that each photo is captured over time. The model "
                     "also predicts how the camera was moving during the photo, and uses that to un-bend the "
                     "geometry inside the model. When the camera is still, nothing changes, so normal photos are "
                     "unaffected by design.", "box")], BOX_BLUE, BLUE))

    # ============================================================ 2. setup
    st.append(PageBreak())
    st.append(P("2. What we used", "h1"))
    st.append(h2_with("2.1 The dataset: TUM Rolling-Shutter (Schubert et al., IROS 2019)", table([
        ["Part", "What it is"],
        ["Two cameras, side by side", "Triggered at exactly the same moment, with the same exposure (2 ms). "
                                      "<b>Camera 1 = normal</b> (whole photo at once). <b>Camera 2 = rolling "
                                      "shutter</b> (30.18 ms from top row to bottom row)."],
        ["Recordings", "10 indoor recordings, about 8 minutes in total, 20 photos per second, 1280×1024 pixels."],
        ["Gyroscope", "Motion sensor: how fast the rig turns, 200 times per second."],
        ["Motion capture", "Ceiling cameras track the rig 120 times per second. This gives the <b>true path</b>, "
                           "our answer key."],
        ["Fairness recording", "One extra recording in which <b>both</b> cameras took normal photos. It tells us "
                               "how different the two cameras are, regardless of rolling shutter."],
    ], [0.27 * W, 0.73 * W])))
    st.append(h2_with("2.2 The three AI models (used as-is: photos only, no hints, no training)", table([
        ["Model", "Notes"],
        ["MapAnything (Meta)", "Apache-licence weights. Originally planned as our main model."],
        ["VGGT-1B (Meta and Oxford)", "Widely used feed-forward 3D model."],
        ["π³ (“Pi3”)", "Newer model (2025). The most accurate of the three on normal photos."],
    ], [0.3 * W, 0.7 * W])))
    st.append(P("Each model receives 16 photos and returns where the camera was for each photo, plus the 3D shape "
                "of the scene.", "small"))
    st.append(h2_with("2.3 Clips and speed groups", table([
        ["Item", "Value"],
        ["Clip", "16 photos, every 3rd photo (2.25 s of walking). A new clip starts every 8 photos."],
        ["Number of clips", "1,139 made, <b>1,124 kept</b>. 15 were dropped because the rig barely moved "
                            "(under 20 cm) or the answer key had gaps. The same clips are used in every "
                            "experiment."],
        ["Speed groups (by gyroscope)", "<b>slow</b>: under 29°/s (0.5 rad/s), 56 clips. <b>medium</b>: 29–86°/s, "
                                        "648 clips. <b>fast</b>: 86°/s or more (1.5 rad/s), 420 clips."],
        ["Turning speed range", "6–206°/s across clips (median 68°/s)."],
    ], [0.27 * W, 0.73 * W])))
    st.append(h2_with("2.4 How we score", table([
        ["Score", "Meaning"],
        ["<b>Path error</b> (main score, “ATE”)", "How far the model's camera positions are from the true ones, on "
                                                  "average, in cm. The guess is first lined up and scaled to the "
                                                  "true path, because the models cannot know real-world size."],
        ["Turning error (“RPE-rot”)", "How wrong the model is about how much the camera turned between "
                                      "neighbouring photos, in degrees."],
        ["Step error (“RPE-trans”)", "How wrong the model is about how far the camera moved between neighbouring "
                                     "photos, in cm."],
        ["Times worse", "Rolling-shutter error ÷ normal error, using the middle value (median) of each group."],
        ["Share of extra error removed", "(bent − straightened) ÷ (bent − normal). 0% = no help. 100% = back to "
                                         "the normal camera's level."],
        ["95% range", "We resampled the clips 1,000 times. The true value lies in this range with 95% "
                      "confidence."],
        ["p", "Chance of seeing a difference this consistent by pure luck. “&lt; 0.001” means almost certainly "
              "real."],
    ], [0.3 * W, 0.7 * W])))

    # ============================================================ 3. EXP0
    st.append(PageBreak())
    st.append(P("3. EXP0: is there a problem?", "h1"))
    st.append(box([P("Do 3D AI models make bigger mistakes on rolling-shutter photos than on normal photos of the "
                     "same moment, and do the mistakes grow when the camera moves faster?", "box")], BOX_GRAY, INK2))
    st.append(P("3.1 What we did", "h2"))
    st += numbered([
        "Prepared both cameras' photos identically: the same lens correction to one shared lens model and the "
        "same brightness curve. That way, only the shutter type differs.",
        "Ran each model on every clip twice: once with camera 1 (normal) and once with camera 2 (rolling "
        "shutter). That is 3 models × 1,124 clips × 2 cameras = 6,744 runs.",
        "Scored every run against the true path, then compared the cameras within each speed group.",
        "Ran the same pipeline on the fairness recording (both cameras normal): 112 clips, split into thirds by "
        "speed.",
    ])
    st.append(h2_with("3.2 Main result: how much worse on the rolling-shutter camera", fig(
        "f3_how_much_worse.png", "Figure 3. Times worse (path error) on the rolling-shutter camera, by speed group. "
                                 "Thin lines show the 95% range. Dashed line: the 1.5× GO line.")))
    st.append(h2_with("Table 3.1. Path error (cm) by model and speed", t_e0_metric("ATE", "cm")))
    st.append(P("“Clips worse on RS” = share of clips where the rolling-shutter run had the larger error.", "small"))
    st += bullets([
        f"<b>π³</b> gets steadily worse as speed rises: {e0('pi3', 'slow', 'ATE').ratio:.2f}× → "
        f"{e0('pi3', 'medium', 'ATE').ratio:.2f}× → <b>{fast['pi3'].ratio:.2f}×</b>.",
        "<b>VGGT</b> is about 1.5–1.7× worse at every speed.",
        "<b>MapAnything</b> is only 1.06–1.16× worse. It is also 2–6× less accurate than the others on normal "
        "photos, which can hide the effect.",
    ])

    st.append(h2_with("3.3 Error against speed, in centimetres", fig(
        "f4_error_vs_speed.png", "Figure 4. Typical path error against turning speed. Blue = normal camera, orange = "
                                 "rolling shutter. The shaded area is the extra error.")))
    st.append(h2_with("Table 3.2. Finer speed steps (path error)", t_e0_fine()))
    st.append(P(f"For π³, the extra error grows from +{pf.RS_minus_GS_cm.iloc[0]:.1f} cm at the slowest speeds to "
                f"+{pf.RS_minus_GS_cm.iloc[-1]:.1f} cm at the fastest ({pf.ratio.iloc[-1]:.1f}× worse).", "small"))

    st.append(h2_with("3.4 The other two scores (Tables 3.3 and 3.4)", t_e0_metric("RPE_rot", "deg")))
    st.append(P("Table 3.3. Turning error (degrees between neighbouring photos). For this score, all three models "
                "get steadily worse as speed rises.", "cap"))
    st.append(t_e0_metric("RPE_trans", "cm"))
    st.append(P("Table 3.4. Step error (cm between neighbouring photos).", "cap"))

    st.append(h2_with("3.5 One example: the fastest clip", fig(
        "f5_fastest_clip.png", "Figure 5. Grey = true path. Coloured dots = where π³ thinks the camera was for each "
                               "of the 16 photos. Normal photos: 2.8 cm off. Rolling-shutter photos: 8.2 cm off.")))

    st.append(h2_with("3.6 Fairness check: is camera 2 just a worse camera?",
                      P("Camera 2 has its own lens, sensor and position. In the fairness recording, <b>both cameras "
                        "take normal photos</b>. If camera 2 were simply worse, it would lose there too."),
                      fig("f6_fairness.png", "Figure 6. Grey: both cameras normal (all clips). Orange: the real "
                                             "test, fast turning.")))
    st.append(h2_with("Table 3.5. Fairness recording, path error (both cameras in normal mode)", t_e0_control()))
    st += bullets([
        "Camera 2 is at most 1.0–1.2× worse when both cameras are normal. That explains only a small part of the "
        "1.5–2.6× seen in the real test.",
        "Divided by the fairness ratio, the fast-motion penalty is still 1.17× (MapAnything), 1.25× (VGGT) and "
        "<b>2.33×</b> (π³).",
        "Caveat: the fairness recording films a calibration board, not the room, so it is a good check but not "
        "a perfect one.",
    ])

    st.append(P("3.7 Other checks", "h2"))
    st += bullets([
        "<b>Timing of the answer key.</b> A rolling-shutter photo has no single capture time. We used the middle "
        "row's time (15 ms after the first row). Using the first row's time instead makes rolling shutter look "
        "worse still: +1.6% (MapAnything), +2.6% (VGGT), +10% (π³). So our choice is the cautious one.",
        "<b>Turning vs walking.</b> In this dataset, fast-turning clips also tend to be fast-walking clips "
        "(correlation 0.40). Even “slow” clips walk at about 0.6 m/s and still have about 7.5 pixels of bend "
        "from turning. Extra statistics are in Appendix D.",
        "<b>Not a factor:</b> blur (same 2 ms exposure on both cameras), timing (photos paired exactly, 0.0 ms "
        "offset) and preparation (identical for both cameras).",
    ])

    st.append(KeepTogether([P("3.8 The decision (pre-agreed rules)", "h2"), table([
        ["Result", "Rule"],
        ["<font color='#1a7f37'><b>GO</b></font>", "Fast motion ≥ 1.5× worse, AND it grows with speed, AND slow "
                                                   "motion ≤ 1.2×. Must hold for MapAnything plus one other model."],
        ["<b>GRAY</b>", "In between: run EXP1 and decide there."],
        ["<font color='#c4321b'><b>STOP</b></font>", "Under 1.2× in fast motion for all models: drop as a main "
                                                     "paper."],
    ], [0.15 * W, 0.85 * W]), Spacer(1, 6), table([
        ["Model", "Fast motion", "Grows with speed?", "Slow motion ≤ 1.2×?", "Result"],
        ["MapAnything", f"{fast['mapanything'].ratio:.2f}×", "barely", "yes (1.14×)", "STOP"],
        ["VGGT", f"{fast['vggt'].ratio:.2f}×", "no (turning error: yes)", "no (1.73×)", "GRAY"],
        ["π³", f"<b>{fast['pi3'].ratio:.2f}×</b>", "<b>yes, clearly</b>", "no (1.70×)", "GRAY"],
    ], [0.18 * W, 0.15 * W, 0.25 * W, 0.24 * W, 0.18 * W]), Spacer(1, 6),
        box([P("<b>EXP0 overall: GRAY.</b> The effect is large for VGGT and π³, but even slow clips were worse, so "
               "EXP0 alone could not prove that the bending is the cause. EXP1 was designed to settle that.", "box")],
            BOX_ORANGE, ORANGE)]))
    st.append(P("3.9 What EXP0 proved, and what it could not", "h2"))
    st += bullets([
        "<b>Proved:</b> rolling shutter makes VGGT and π³ clearly worse. For π³ it gets worse with speed. It is "
        "not just “camera 2 is worse”.",
        "<b>Could not prove:</b> that the bending is the real cause (the slow-clip puzzle), and how much accuracy "
        "we could win back by fixing it.",
    ])

    # ============================================================ 4. EXP1
    st.append(PageBreak())
    st.append(P("4. EXP1: can it be fixed if we know the motion?", "h1"))
    st.append(box([P("If we straighten every rolling-shutter photo using the <b>true</b> camera motion, how much of "
                     "the extra error goes away? <b>Pre-agreed rule:</b> GO if at least 50% goes away in fast "
                     "motion. STOP if under 25%.", "box")], BOX_GRAY, INK2))
    st.append(P("4.1 How straightening works", "h2"))
    st.append(fig("g1_unroll_concept.png", "Figure 7. Straightening, step by step. Every row is moved to where it "
                                           "would have been at the moment the middle row was captured."))
    st += numbered([
        "<b>E1a, the planned test (turning only):</b> the gyroscope says how fast the camera was turning during "
        "the 30 ms. From that we compute, for every row, how far it moved, and slide it back. No depth needed.",
        "<b>E1b, the extra test we recommended after EXP0 (turning + walking):</b> walking also bends the photo, "
        "but how much depends on how far away each object is. Near things shift more. So E1b also needs distances. "
        "We used π³'s depth estimate, scaled to metres with the true path, plus the true walking speed.",
        "The straightening is done on the raw lens image, before the usual lens correction. That keeps each row's "
        "timing exact. The rest of the preparation is identical to EXP0.",
        "All three models are re-run on the straightened photos: same 1,124 clips, same scoring, same answer key.",
    ])
    st.append(fig("g2_unroll_photos.png", "Figure 8. The fastest frame (about 180°/s): normal camera, bent "
                                          "rolling-shutter photo, and straightened. Look at the door frame and the "
                                          "cupboard edge. This check picture was made before we switched how edges "
                                          "are filled, so a thin black sliver shows at the bottom right. In the real "
                                          "run, pixels from beyond the sensor edge repeat the edge pixel instead."))
    st.append(h2_with("4.2 Safety checks (all passed before the run)", table([
        ["Check", "Result"],
        ["With zero motion, is the photo unchanged?", "Yes: identical to EXP0's photo (0.0 pixel difference)."],
        ["Are the gyroscope's directions right?", "The predicted rotation over one photo matches the true rotation "
                                                  "within 0.13° (out of 4.45°). A sign mistake would show up as "
                                                  "8.96°."],
        ["How big are the shifts?", "Up to 35 pixels, at 3.1 rad/s (about 180°/s)."],
        ["Does it look right?", "Yes: the straightened photo lines up with the normal camera's (Figure 8)."],
    ], [0.38 * W, 0.62 * W])))

    st.append(h2_with("4.3 Main result: share of the extra error removed", fig(
        "g3_gap_closed.png", "Figure 9. Share of the extra rolling-shutter error removed, by speed group. Thin lines "
                             "show the 95% range. MapAnything's slow-group bars are noise: its gap there is only "
                             "0.5 cm.")))
    st.append(h2_with("Table 4.1. Path error (cm) and share of the extra error removed", t_e1_metric("ATE", "cm")))
    st.append(P("Values above 100% mean the straightened photos scored slightly better than the normal camera. "
                "These differences are within the camera-to-camera spread measured by the fairness check, so read "
                "them as “back to normal level”.", "small"))

    st.append(h2_with("4.4 The same thing in centimetres", fig(
        "g4_error_cm.png", "Figure 10. Typical path error for each version of the photos.")))
    st.append(h2_with("4.5 Does it work at every speed?", fig(
        "g5_gap_vs_speed.png", "Figure 11. Share removed against turning speed (six equal-size groups of clips per "
                               "model). The share is flat: the fix works as well for slow clips as for fast ones.")))
    st.append(h2_with("4.6 The fastest clip again", fig(
        "g6_fastest_clip_e1.png", "Figure 12. The same fastest clip as Figure 5. π³: 2.8 cm (normal), 8.2 cm (bent), "
                                  "2.7 cm (straightened for turning), 2.5 cm (turning + walking)."), t_fastest()))

    st.append(h2_with("4.7 How consistent is the improvement? (Table 4.2)", t_e1_consistency()))
    st.append(P("“Improved” = share of clips where the straightened photos gave a smaller path error than the bent "
                "ones. p compares straightened against bent, clip by clip.", "small"))

    st.append(h2_with("4.8 The other two scores (Tables 4.3 and 4.4)", t_e1_metric("RPE_rot", "°")))
    st.append(P("Table 4.3. Turning error (degrees). “No gap” = rolling shutter was within 5% of normal, so there "
                "was nothing to remove.", "cap"))
    st.append(t_e1_metric("RPE_trans", "cm"))
    st.append(P("Table 4.4. Step error (cm).", "cap"))

    st.append(h2_with("4.9 The decision", table([
        ["Model", "Fast clips: turning only (planned test)", "Fast clips: turning + walking", "Result"],
        *[[NAME[m], f"<b>{g_rsu[m].recovered * 100:.0f}%</b> [{g_rsu[m].recovered_ci_lo * 100:.0f}, "
                    f"{g_rsu[m].recovered_ci_hi * 100:.0f}]",
           f"{g_rsut[m].recovered * 100:.0f}% [{g_rsut[m].recovered_ci_lo * 100:.0f}, "
           f"{g_rsut[m].recovered_ci_hi * 100:.0f}]", "<font color='#1a7f37'><b>GO</b></font>"] for m in MODELS],
    ], [0.18 * W, 0.32 * W, 0.32 * W, 0.18 * W]), Spacer(1, 6),
        box([P("<b>EXP1 overall: GO</b> for all three models, in both versions. Even the bottom of every 95% range "
               "is well above the 50% GO line.", "box")], BOX_GREEN, GREEN)))
    st.append(P("4.10 What EXP1 proved, and what it did not", "h2"))
    st += bullets([
        "<b>Proved: the bending is the whole cause.</b> Once the photos are straightened, accuracy goes back to the "
        "normal camera's level, at every speed.",
        "<b>Solved EXP0's slow-clip puzzle.</b> The slow-clip penalty was also rolling shutter, mostly from walking: "
        "slow-clip share removed for π³ 86% → 109%, VGGT 63% → 83% when walking is added.",
        "<b>Walking matters, not just turning.</b> Adding walking gains 15–20 points (π³ fast 82% → 98%). Our "
        "method should predict both.",
        "<b>Did not prove that a real system can do this.</b> E1a used a real gyroscope. E1b used the true path "
        "and true scale.",
        "<b>Did not prove it generalizes.</b> One room, one camera rig, one readout time (30 ms).",
    ])

    # ============================================================ 5. where we stand
    st.append(PageBreak())
    st.append(P("5. Where we stand", "h1"))
    st.append(table([
        ["We now know (measured)", "We don't know yet"],
        ["Rolling shutter hurts VGGT and π³ a lot (1.5–2.6× in fast motion) and MapAnything a little (1.16×).",
         "Can a model estimate the motion from images well enough to fix itself? <b>(EXP2)</b>"],
        [f"The damage grows with speed (π³: {pf.ratio.iloc[0]:.1f}× → {pf.ratio.iloc[-1]:.1f}×).",
         "Does a trained “motion head” beat simple fixes? <b>(EXP3)</b>"],
        ["It is caused by rolling shutter, not camera differences (those are at most 1.2×).",
         "Does it hold on phones, other readout times and outdoor scenes?"],
        ["Knowing the motion removes 82–100% of the damage, at every speed.",
         "Has someone already published this exact idea? <b>(novelty check)</b>"],
        ["Both turning and walking matter.", ""],
    ], [0.5 * W, 0.5 * W]))

    # ============================================================ 6. road ahead
    st.append(P("6. The road ahead, and why", "h1"))
    st.append(fig("g7_roadmap.png", "Figure 13. The plan from here. Green = done, amber = next, grey = later. The "
                                    "decision lines (70% / 40%) are a proposal for the team to confirm before EXP2 "
                                    "runs.", width=0.86 * W))
    st.append(P("Deadline", "h2"))
    st.append(P("CVPR 2027: registration around <b>10 November</b> and submission around <b>16 November 2026</b> "
                "(from the project brief; please verify on the CVPR site). As of 29 October that is about "
                "<b>12 days to registration and 18 days to submission</b>. It is tight but workable, since each "
                "experiment so far took about 2 hours of computer time."))
    st.append(P("Step 1: prior-work check (now, alongside EXP2)", "h2"))
    st += bullets([
        "<b>What:</b> search recent arXiv (last 60 days) and the main venues for rolling shutter combined with "
        "feed-forward 3D models. Also search for gyroscope-based rolling-shutter correction, SMARSC (CVPR 2019) "
        "and rolling-shutter unrolling networks.",
        "<b>Why:</b> similar adapters for fisheye lenses were published recently (RayTun3R, Fisheye3R), so rolling "
        "shutter is the obvious next target and scoop risk is real. Reviewers will also ask “why not just use the "
        "gyroscope?”, so we must know exactly what already exists.",
    ])
    st.append(P("Step 2: EXP2, the cheap training-free fix (next; projected 3–5 hours including coding)", "h2"))
    st += numbered([
        "Run the model once on the bent clip.",
        "From its own answer, estimate how the camera was turning and moving: compare neighbouring photos' "
        "positions and divide by the time between them.",
        "Straighten the photos with that estimate and run the model again.",
        "Compare against EXP1 (the best case) and against the bent photos.",
    ])
    st += bullets([
        "<b>Why it is next:</b> it decides which paper we write, it answers the “no gyroscope” question directly, "
        "and it costs very little because EXP1's code is reused. We need it as a comparison in the final paper "
        "anyway.",
        "<b>A helpful detail:</b> the bend from walking depends on speed ÷ distance. The model gives both in the "
        "same unknown units, so the units cancel and EXP2 can correct walking too, without knowing real-world "
        "scale.",
        "<b>Expected difficulty:</b> neighbouring photos in a clip are 150 ms apart, while a photo takes 30 ms, "
        "and the first-pass answer is itself bent. So the estimate will be rougher than the gyroscope's. EXP2 "
        "measures how much that matters.",
        "<b>Decision lines (proposal, agree before running):</b> “close to EXP1” = EXP2 achieves at least 70% of "
        "EXP1's gain in fast clips. “Far below” = under 40%. In between: report EXP2 as a strong baseline and also "
        "build EXP3.",
    ])
    st.append(P("Step 3: depends on EXP2", "h2"))
    st += bullets([
        "<b>If EXP2 is close to EXP1, Paper A:</b> a training-free paper, “two passes fix rolling shutter in any 3D "
        "model”. Cheaper and lower-risk.",
        "<b>If EXP2 falls short, EXP3:</b> build a generator of synthetic bent videos (from ScanNet++ / TartanAir, "
        "which have depth and camera paths). Train a small motion head plus a light adapter (LoRA) that predicts "
        "turning and walking per photo. GO only if it clearly beats the same adapter trained <i>without</i> the "
        "motion head, and does not hurt normal photos. Projected 4–6 h for the first run, plus building the "
        "generator.",
    ])
    st.append(P("Step 4: full benchmark and writing", "h2"))
    st.append(P("Compare: plain model; “straighten the photo with an existing unrolling network, then "
                "reconstruct”; gyroscope straightening (E1a, already done); EXP2; adapter only; ours. Measure path "
                "error on TUM, 3D shape quality on synthetic rolling-shutter data (as far as we know, TUM has camera "
                "paths only, no 3D shape ground truth) and no loss on normal photos. Teaser figure: a fast pan where "
                "the plain model bends a wall and ours keeps it straight."))
    st.append(KeepTogether([P("Recommended changes to the original plan", "h2"), table([
        ["Change", "Why (evidence)"],
        ["<b>Main model: π³, then VGGT.</b> MapAnything becomes secondary (the plan had it first).",
         "π³ has the largest gap (2.6×) and the tightest results ([79, 85]%). MapAnything's gap is so small that "
         "gains are hard to show."],
        ["<b>Predict both turning and walking</b>, kept small by regularization (the plan said “focus on "
         "turning”).", "E1b: walking adds 15–20 points and explains the slow-clip penalty."],
        ["<b>Add gyroscope straightening as a baseline.</b>", "E1a shows it works very well. Reviewers will ask, and "
                                                              "we already have the numbers."],
        ["<b>Speed up the pipeline before EXP2.</b>", "Most run time was spent on the CPU (straightening, loading "
                                                      "photos), not the GPU. <i>Projected (untested):</i> caching "
                                                      "and GPU straightening could cut run time by half or more."],
    ], [0.45 * W, 0.55 * W])]))
    st.append(KeepTogether([P("Risks", "h2"), table([
        ["Risk", "How likely", "What we do"],
        ["Someone publishes the same idea first", "Medium (fisheye versions are out)",
         "Re-search arXiv weekly, move fast, keep the workshop as a fallback."],
        ["“Just use a gyroscope”", "High, as a reviewer question",
         "Images-only is our angle. Include the gyroscope baseline. Stress videos with no sensor data."],
        ["The model's own motion estimate is too rough", "Unknown", "EXP2 tests exactly this. If too rough, a "
                                                                    "learned motion head is the paper (EXP3)."],
        ["Only one dataset so far", "Certain right now", "Synthetic rolling-shutter data with different readout "
                                                         "times. A second real dataset if one exists."],
        ["Walking speed is hard to estimate", "Medium", "Regularize toward zero. Report turning-only and "
                                                        "turning + walking separately."],
        ["Deadline", "Tight (~18 days)", "Each step has a GO/STOP rule. The workshop track is the fallback."],
    ], [0.3 * W, 0.22 * W, 0.48 * W])]))

    # ============================================================ appendices
    st.append(PageBreak())
    st.append(P("Appendix A. Glossary", "h1"))
    st.append(table([
        ["Term", "Meaning"],
        ["Rolling shutter (RS)", "Camera reads the photo row by row (here over 30.18 ms)."],
        ["Global shutter (GS), “normal”", "Camera captures every row at the same instant."],
        ["Clip (window)", "16 photos from 2.25 s of a recording."],
        ["Path error (ATE)", "Average distance between the model's camera positions and the true ones, after "
                             "lining up and scaling (cm)."],
        ["Turning error (RPE-rot)", "Error in how much the camera turned between neighbouring photos (degrees)."],
        ["Step error (RPE-trans)", "Error in how far the camera moved between neighbouring photos (cm)."],
        ["Straightening (unrolling)", "Moving each row of a rolling-shutter photo to where it would be if the "
                                      "whole photo had been taken at once."],
        ["Oracle", "Using information a real system would not have (here: gyroscope and true path), to measure the "
                   "best case."],
        ["Share of extra error removed", "(bent − straightened) ÷ (bent − normal), using medians."],
        ["Fairness-adjusted share", "The same, but measured against the normal camera's error × the fairness "
                                    "ratio (how much worse camera 2 is when both cameras are normal)."],
        ["95% range", "Bootstrap confidence interval: clips resampled 1,000 times."],
        ["p (Wilcoxon test)", "Clip-by-clip test of whether one version is consistently better than the other."],
        ["rad/s", "Radians per second. 1 rad/s ≈ 57°/s."],
        ["LoRA / adapter", "A small add-on trained on top of a frozen model, so the big model is not retrained."],
    ], [0.3 * W, 0.7 * W]))

    st.append(P("Appendix B. Run facts", "h1"))
    st.append(table([
        ["", "EXP0", "EXP1"],
        ["Total time", "about 1 h 50 min (setup 25 min, photo preparation 5 min, model runs about 70 min, analysis "
                       "5 min)", "2 h 02 min (17:14 to 19:15 UTC)"],
        ["Photo work", "1,124 clips × 2 cameras, plus 112 fairness clips",
         "9,248 photos straightened per version (turning: 14 min; turning + walking: 17 min, on the CPU). π³ depth "
         "for E1b: 65 min."],
        ["Model run time", "MapAnything 58 min, VGGT about 70 min, π³ about 78 min (2,248 clips each, side by side)",
         "Turning: MapAnything 39.0, VGGT 46.1, π³ 48.2 min. Turning + walking: 31.9, 37.6, 39.0 min (1,124 clips "
         "each, side by side)."],
        ["GPU memory (peak per model)", "MapAnything 13.5 GB, VGGT 7.5 GB, π³ 6.1 GB",
         "same as EXP0"],
        ["Shared GPU (80 GB)", "Peak use 38.7 GB. Free memory never below about 42 GB.",
         "Free memory never below 26.9 GB (rule: 12 GB). Our processes were set to low CPU priority so the other "
         "team's job kept first pick."],
        ["Settings", "bf16 precision, one clip at a time per model, photos only", "same as EXP0"],
    ], [0.2 * W, 0.4 * W, 0.4 * W]))

    st.append(P("Appendix C. Changes from the plan", "h1"))
    st.append(P("EXP0", "h2"))
    st += bullets([
        "π³'s code moved to a new address (github.com/yyfz/Pi3). The same official code and weights were used.",
        "The three models ran side by side to save time. MapAnything's optional output masks were turned off for "
        "speed; this does not affect camera positions.",
        "Extras not in the plan: the fairness recording, finer speed steps, the timing-sensitivity check, and the "
        "turning/walking statistics.",
    ])
    st.append(P("EXP1", "h2"))
    st += bullets([
        "Straightening is done on the raw lens image, not with the plan's simple formula on a corrected image. "
        "It is the same idea, but exact for this fisheye-like lens.",
        "Turning comes from the gyroscope (averaged over the 30 ms of each photo). The true path was used only to "
        "check it.",
        "Added E1b (turning + walking). It uses the true walking speed and π³'s depth scaled with the true path. "
        "π³'s depth was used for all three models, which may slightly favour π³ in E1b.",
        "Pixels shifted in from beyond the sensor edge repeat the edge pixel instead of being black, so a black "
        "border cannot give the models a hint. At fast motion this leaves smeared strips up to about 35 pixels "
        "wide. That can only hurt the straightened result, so the shares removed are, if anything, conservative.",
    ])

    st.append(P("Appendix D. Extra statistics", "h1"))
    st.append(h2_with("D.1 Does the penalty grow with turning speed? (EXP0, all 1,124 clips)", t_trend()))
    st.append(P("Positive ρ = the rolling-shutter penalty grows with turning speed. π³ and MapAnything: yes for path "
                "and turning error. VGGT: yes for turning error only.", "small"))
    st.append(h2_with("D.2 Turning vs walking: which drives the penalty? (EXP0)", t_reg()))
    st.append(P("A straight-line fit of log(times worse) on turning speed and walking speed together. Turning speed "
                "and walking speed are correlated (0.40), so read these as rough. For π³, turning clearly drives "
                "the penalty.", "small"))
    st.append(h2_with("D.3 Direction-of-travel accuracy (EXP0): share of photo pairs with direction off by "
                      "less than 5°", t_e0_rta()))
    rta = e0("pi3", "fast", "RTA5")
    st.append(P(f"Higher is better. For π³ in fast motion it drops from {rta.median_GS * 100:.0f}% to "
                f"{rta.median_RS * 100:.0f}% on rolling-shutter photos. Turning accuracy (the same test for "
                "rotation) had a median of 97–100% on both cameras for every model and speed, so it is not shown.",
                "small"))
    st.append(h2_with("D.4 EXP1 share removed, plain vs fairness-adjusted (path error)", t_e1_floor()))
    st.append(P("Fairness-adjusted: the target is the normal camera's error × the fairness ratio (1.00 / 1.18 / "
                "1.12), because camera 2 is slightly worse even without rolling shutter. MapAnything's slow "
                "numbers are noise (tiny gap).", "small"))

    st.append(PageBreak())
    st.append(P("Appendix E. Technical charts (as produced by the analysis code)", "h1"))
    st.append(ext_fig(f"{RES0}/ratio_vs_speed.png", "E.1 EXP0: times worse against turning speed, six equal-size "
                                                    "speed groups, with 95% ranges."))
    st.append(ext_fig(f"{RES0}/qual_traj_s9_w0077.png", "E.2 EXP0: top-down camera paths for the fastest clip, all "
                                                        "three models."))
    st.append(ext_fig(f"{RES1}/e1_ate_bars.png", "E.3 EXP1: median path error per speed group for normal, bent, "
                                                 "straightened (turning) and straightened (turning + walking)."))
    st.append(ext_fig(f"{RES1}/e1_recovered_vs_speed.png", "E.4 EXP1: share removed against turning speed, both "
                                                           "versions, all models."))

    st.append(P("Appendix F. Files", "h1"))
    st.append(table([
        ["Where", "What"],
        ["<font face='Courier'>EXPERIMENTS_SO_FAR.md</font>", "Short written summary of both experiments and the road "
                                                              "ahead."],
        ["<font face='Courier'>results/exp0/</font>", "summary.md, DATA_NOTES.md, all tables (CSV) and charts."],
        ["<font face='Courier'>results/exp1/</font>", "summary.md, e1_summary.csv, per_window_e1.csv (every clip), "
                                                      "charts."],
        ["<font face='Courier'>src/</font>", "All code: preparation, model runs, straightening, analysis."],
        ["<font face='Courier'>report/</font>", "This report, the earlier EXP0 report, and the scripts that build "
                                                "them."],
        ["Only on the GPU box", "EXP1's raw model outputs, straightened photos and depth maps. They can be rebuilt "
                                "in about 2 hours. Every score is already saved locally."],
    ], [0.35 * W, 0.65 * W]))

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="RS-3D: Experiments 0 and 1, full results",
                            author="RS-3D team", subject="EXP0 gap test and EXP1 oracle unrolling results")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
