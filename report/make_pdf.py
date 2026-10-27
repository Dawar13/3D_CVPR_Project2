"""Builds the plain-language EXP0 team report: report/EXP0_Rolling_Shutter_Report.pdf"""
import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "EXP0_Rolling_Shutter_Report.pdf")

FONT_DIR = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("Arial", f"{FONT_DIR}/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", f"{FONT_DIR}/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{FONT_DIR}/ariali.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
                              boldItalic="Arial-Bold")

INK = colors.HexColor("#1a1a19")
INK2 = colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6")
ORANGE = colors.HexColor("#eb6834")
GRID = colors.HexColor("#d9d8d0")
BOX_BLUE = colors.HexColor("#eef4fc")
BOX_ORANGE = colors.HexColor("#fdf0ea")
BOX_GRAY = colors.HexColor("#f4f4f1")
HEAD_BG = colors.HexColor("#e9eef6")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
FRAME_W = PAGE_W - 2 * MARGIN

S = dict(
    title=ParagraphStyle("title", fontName="Arial-Bold", fontSize=21, leading=26, textColor=INK, spaceAfter=4),
    sub=ParagraphStyle("sub", fontName="Arial", fontSize=10.5, leading=14, textColor=INK2, spaceAfter=10),
    h1=ParagraphStyle("h1", fontName="Arial-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=12,
                      spaceAfter=6),
    h2=ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12, leading=16, textColor=INK, spaceBefore=8,
                      spaceAfter=4),
    body=ParagraphStyle("body", fontName="Arial", fontSize=10.5, leading=15, textColor=INK, spaceAfter=6,
                        alignment=TA_LEFT),
    bullet=ParagraphStyle("bullet", fontName="Arial", fontSize=10.5, leading=15, textColor=INK, leftIndent=14,
                          bulletIndent=3, spaceAfter=3),
    cap=ParagraphStyle("cap", fontName="Arial-Italic", fontSize=9, leading=12, textColor=INK2, spaceBefore=3,
                       spaceAfter=10),
    cell=ParagraphStyle("cell", fontName="Arial", fontSize=9.5, leading=12.5, textColor=INK),
    cellb=ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=9.5, leading=12.5, textColor=INK),
    small=ParagraphStyle("small", fontName="Arial", fontSize=9, leading=12, textColor=INK2, spaceAfter=4),
    box=ParagraphStyle("box", fontName="Arial", fontSize=10.5, leading=15, textColor=INK, spaceAfter=4),
    boxh=ParagraphStyle("boxh", fontName="Arial-Bold", fontSize=12, leading=16, textColor=INK, spaceAfter=4),
)


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def numbered(items):
    return [Paragraph(t, S["bullet"], bulletText=f"{i}.") for i, t in enumerate(items, 1)]


def fig(name, caption, width=FRAME_W):
    path = os.path.join(FIG, name)
    w, h = PILImage.open(path).size
    img = Image(path, width=width, height=width * h / w)
    return KeepTogether([img, P(caption, "cap")])


def box(flowables, bg=BOX_BLUE, edge=BLUE):
    t = Table([[flowables]], colWidths=[FRAME_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if (header and i == 0) else "cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafaf8")))
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8.5)
    canvas.setFillColor(INK2)
    canvas.drawString(MARGIN, 10 * mm, "RS-3D  ·  EXP0 gap test  ·  all numbers measured on 27 Oct 2026")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


def build():
    st = []
    W = FRAME_W

    # ------------------------------------------------------------------ page 1
    st += [P("EXP0: Do 3D AI models break on rolling-shutter cameras?", "title"),
           P("RS-3D project &nbsp;·&nbsp; Experiment 0, the gap test &nbsp;·&nbsp; run on 27 October 2026 "
             "&nbsp;·&nbsp; for Dawar, Shubh and Sujal", "sub")]
    st.append(box([
        P("The answer in one minute", "boxh"),
        *bullets([
            "<b>Yes, mostly.</b> Two of the three models make clearly bigger mistakes on rolling-shutter photos. "
            "<b>π³ is up to 2.6× worse</b> when the camera turns fast, and the faster it turns, the worse it gets.",
            "<b>MapAnything is barely affected</b>: only about 1.2× worse.",
            "Our pre-agreed rule gives <b>GRAY</b>, not GO. The effect is strong, but one check did not come out "
            "clean: even slowly moving clips were worse.",
            "<b>Next: EXP1.</b> We will “un-bend” the photos using the true camera motion and test again. That "
            "shows whether the bending really is the cause, and how much we could win by fixing it.",
        ]),
    ]))
    st.append(Spacer(1, 6))
    st.append(P("Every number in this report was <b>measured</b> in the experiment. None are guesses or "
                "projections.", "small"))

    st.append(P("1. The problem in one picture", "h1"))
    st.append(P("Most cameras in phones, drones and robots do not take the whole photo at once. They read it "
                "<b>row by row, top to bottom</b>, over about 30 milliseconds. This is called a "
                "<b>rolling shutter</b>. If the camera turns during those 30 ms, straight things come out "
                "slanted or wobbly (the “jello effect”)."))
    st.append(P("3D AI models assume every photo was taken in a single instant. EXP0 checks whether that wrong "
                "assumption actually hurts them."))
    st.append(fig("f1_sketch.png", "Figure 1. Sketch: a straight pole photographed while the camera turns. "
                                   "Left: all rows at once. Right: one row at a time."))

    # ------------------------------------------------------------------ page 2
    st.append(fig("f2_photos.png",
                  "Figure 2. Real photos from our dataset, taken at the same instant while the camera turned at "
                  "about 200° per second. At this speed the rows of the right photo are shifted by up to "
                  "~70 pixels (calculated from the measured turning speed). Part of the visible difference is "
                  "also because the two cameras sit 11 cm apart."))

    st.append(P("2. What EXP0 asked", "h1"))
    st.append(box([P("Do 3D AI models make bigger mistakes on rolling-shutter photos than on normal photos, and "
                     "do the mistakes grow when the camera moves faster?", "box")], BOX_GRAY, INK2))
    st.append(Spacer(1, 6))
    st.append(P("<b>Why it matters:</b> if yes, our project idea (teach the model that each row was taken at a "
                "slightly different moment) is worth building. If no, we drop it now and save weeks."))

    st.append(P("3. What we used", "h1"))
    st.append(P("The dataset", "h2"))
    st.append(table([
        ["Part", "What it is"],
        ["TUM Rolling-Shutter dataset", "From the Technical University of Munich (Schubert et al., 2019). A person "
                                        "carried a small rig with <b>two identical cameras side by side</b> around "
                                        "a room. 10 recordings, about 8 minutes in total."],
        ["Camera 1 (left)", "<b>Normal camera</b>: takes the whole photo at once."],
        ["Camera 2 (right)", "<b>Rolling-shutter camera</b>: row by row, 30 ms from the top row to the bottom row."],
        ["Gyroscope", "A motion sensor that measures how fast the rig is turning, 200 times per second."],
        ["Motion capture", "Ceiling cameras tracked the rig 120 times per second. This gives the <b>true path</b> "
                           "of each camera: our answer key."],
    ], [0.28 * W, 0.72 * W]))
    st.append(P("Both cameras take a photo at <b>exactly the same moment</b>, with the same exposure time. So the "
                "test is fair: same room, same instant, same movement. The main difference is how each camera "
                "takes its photo.", "small"))

    st.append(P("The three AI models", "h2"))
    st.append(table([
        ["Model", "Notes"],
        ["MapAnything (Meta)", "Was planned as our main model."],
        ["VGGT (Meta and Oxford)", "Widely used 3D model."],
        ["π³ or “Pi3”", "Newer model from 2025."],
    ], [0.35 * W, 0.65 * W]))
    st.append(P("All three do the same job: you give them a handful of photos and they work out "
                "<b>where the camera was</b> for each photo, plus the 3D shape of the scene. We gave them "
                "<b>only the photos</b>, no extra hints.", "small"))

    # ------------------------------------------------------------------ page 3
    st.append(P("4. How we tested", "h1"))
    st += numbered([
        "<b>Clean the photos the same way for both cameras:</b> fix the lens bend and set the brightness. That "
        "way the only thing left that differs is how each photo was taken.",
        "<b>Cut the recordings into 1,124 short clips</b> of 16 photos each, about 2 seconds of walking per clip. "
        "1,139 clips were made and 15 were dropped, because the rig barely moved or the answer key had gaps.",
        "<b>Measure how fast the camera turned</b> in each clip (gyroscope) and sort the clips into three groups. "
        "<b>Slow:</b> under 30° per second, 56 clips. <b>Medium:</b> 30–85°/s, 648 clips. <b>Fast:</b> over "
        "85°/s, 420 clips.",
        "<b>Run every model twice on every clip</b>: once with camera 1's photos, once with camera 2's photos. "
        "That is 3 models × 1,124 clips × 2 cameras = 6,744 runs.",
        "<b>Score each run:</b> compare the model's guessed camera path with the true path. "
        "<b>“Path error”</b> = how far off the guessed camera positions are, on average, in centimetres. "
        "(The guess is first lined up and scaled to the true path, because the models cannot know the exact "
        "real-world size.)",
        "<b>Compare the two cameras:</b> divide camera 2's error by camera 1's error. "
        "1.0× = no difference. 2.0× = twice as bad.",
    ])

    st.append(P("5. Results", "h1"))
    st.append(P("5.1 How much worse is each model on the rolling-shutter camera?", "h2"))
    st.append(fig("f3_how_much_worse.png",
                  "Figure 3. Bars = how many times bigger the path error is on the rolling-shutter camera. Thin "
                  "black lines = the range we are 95% sure the true value lies in. Dashed line = the 1.5× GO line."))
    st += bullets([
        "<b>π³:</b> clearly worse, and it grows with speed: 1.7× → 2.2× → 2.6×.",
        "<b>VGGT:</b> about 1.5–1.7× worse at every speed. It does not clearly grow with speed.",
        "<b>MapAnything:</b> only 1.06–1.16× worse.",
        "For every model and every speed group, the rolling-shutter camera was worse, and the difference is "
        "far too consistent to be luck.",
        "A second score, how much the turning angle between photos is off, also gets worse with speed for "
        "<b>all three</b> models (fast-group ratios 1.26×, 1.68× and 1.80×; table in the appendix).",
    ])

    # ------------------------------------------------------------------ page 4
    st.append(P("5.2 The same thing in centimetres", "h2"))
    st.append(fig("f4_error_vs_speed.png",
                  "Figure 4. Typical path error (middle value of each group of clips) against turning speed. "
                  "Blue = normal camera, orange = rolling-shutter camera. The shaded area is the extra error "
                  "caused by camera 2."))
    st.append(table([
        ["Model", "Speed group", "Normal camera", "Rolling-shutter camera", "Times worse"],
        ["MapAnything", "slow", "3.4 cm", "3.9 cm", "1.14×"],
        ["", "medium", "6.6 cm", "7.0 cm", "1.06×"],
        ["", "fast", "9.3 cm", "10.9 cm", "1.16×"],
        ["VGGT", "slow", "1.4 cm", "2.4 cm", "1.73×"],
        ["", "medium", "2.3 cm", "3.9 cm", "1.69×"],
        ["", "fast", "6.6 cm", "9.7 cm", "1.47×"],
        ["π³", "slow", "0.8 cm", "1.3 cm", "1.70×"],
        ["", "medium", "1.0 cm", "2.3 cm", "2.19×"],
        ["", "fast", "2.0 cm", "5.1 cm", "<b>2.61×</b>"],
    ], [0.2 * W, 0.17 * W, 0.2 * W, 0.25 * W, 0.18 * W]))
    st.append(Spacer(1, 4))
    st.append(P("<b>What stands out:</b> π³ is by far the most accurate model on normal photos, under 2 cm, and "
                "it is the one most hurt by rolling shutter. When π³'s clips are split more finely by speed, the "
                "extra error grows from 0.4 cm when turning slowly to 5.2 cm at the fastest speeds (3.9× worse). "
                "MapAnything already makes 2–12 cm mistakes on normal photos, so the extra rolling-shutter error "
                "is a small part of its total."))

    # ------------------------------------------------------------------ page 5
    st.append(KeepTogether([P("5.3 One example: the fastest clip", "h2"), fig(
        "f5_fastest_clip.png",
        "Figure 5. Grey = true path of the camera. Coloured dots = where π³ thinks the camera was for each of the "
        "16 photos. On normal photos π³ follows the true path closely; on rolling-shutter photos it drifts off.")]))

    st.append(P("6. Fairness check: is camera 2 just a worse camera?", "h1"))
    st.append(P("<b>The worry:</b> camera 2 sits in a different spot and has its own lens and sensor. Maybe it "
                "is just slightly worse, and row-by-row capture has nothing to do with it."))
    st.append(P("<b>The check (extra, not in the original plan):</b> the dataset has one more recording in which "
                "<b>both cameras took normal photos</b>. We ran the same test on it. If camera 2 were simply "
                "worse, it would still lose clearly there."))
    st.append(fig("f6_fairness.png",
                  "Figure 6. Grey = both cameras in normal mode. Orange = the real test, fast turning. The grey "
                  "bars stay close to 1.0×."))
    st.append(P("<b>Result:</b> with both cameras in normal mode, camera 2 was only 1.00–1.18× worse. So "
                "“camera 2 is just worse” explains a small part at most. One caveat: that recording films a "
                "calibration board, not the room, so it is a good check but not a perfect one."))

    # ------------------------------------------------------------------ page 6
    st.append(KeepTogether([P("7. The decision", "h1"),
                            P("Before running anything we agreed on these rules (from our project plan):"), table([
        ["Result", "Rule"],
        ["<font color='#1a7f37'><b>GO</b></font>", "In fast motion at least 1.5× worse, AND it gets worse as speed "
                                                   "rises, AND slow motion shows almost no difference (at most "
                                                   "1.2×). This must hold for MapAnything plus one other model."],
        ["<b>GRAY</b>", "Fast motion 1.2–1.5× worse, or GO not fully met: run EXP1 and decide there."],
        ["<font color='#c4321b'><b>STOP</b></font>", "Under 1.2× worse in fast motion for all models: drop the "
                                                     "idea as a main paper."],
    ], [0.15 * W, 0.85 * W])]))
    st.append(Spacer(1, 6))
    st.append(table([
        ["Model", "Fast motion", "Grows with speed?", "Slow motion ≤ 1.2×?", "Result"],
        ["MapAnything", "1.16×", "barely", "yes (1.14×)", "STOP"],
        ["VGGT", "1.47×", "no", "no (1.73×)", "GRAY"],
        ["π³", "<b>2.61×</b>", "<b>yes, clearly</b>", "no (1.70×)", "GRAY"],
    ], [0.2 * W, 0.17 * W, 0.22 * W, 0.23 * W, 0.18 * W]))
    st.append(Spacer(1, 6))
    st.append(box([P("<b>Overall: GRAY.</b> GO needs MapAnything to pass, and it did not. It is not STOP either, "
                     "because VGGT and π³ are well above 1.2×. π³ missed GO only because its slow clips were also "
                     "worse (1.70×, where the rule wants 1.2× or less).", "box")], BOX_ORANGE, ORANGE))

    st.append(P("8. What EXP0 proved, and what it could not", "h1"))
    st.append(P("Proved", "h2"))
    st += bullets([
        "Rolling-shutter photos make <b>VGGT and π³ clearly worse</b>: 1.5–2.6× in fast motion.",
        "For π³, <b>the faster the camera turns, the worse it gets</b>, exactly the pattern our idea predicts.",
        "It is <b>not just</b> “camera 2 is a worse camera”: the fairness check gives 1.2× at most.",
        "<b>MapAnything is barely affected</b>, partly because its own mistakes are already large.",
    ])
    st.append(P("Could not prove", "h2"))
    st += numbered([
        "<b>That the bending is really the cause.</b> Even slow clips were about 1.7× worse for VGGT and π³. "
        "Two possible reasons, and EXP0 cannot tell them apart. "
        "<b>(a)</b> The slow clips still have some bending: the rig still turns a little (about 7 pixels of "
        "bend) and walks at about 0.6 m per second, and walking also bends rolling-shutter photos. "
        "<b>(b)</b> Something else about camera 2 in this room.",
        "<b>How much we could gain by fixing it.</b> Knowing the models are worse does not tell us how much "
        "better they get once the bending is removed. That is the size of the prize.",
        "<b>Whether a cheap fix is already enough.</b> That is EXP2's job.",
    ])

    # ------------------------------------------------------------------ page 7
    st.append(P("9. What EXP1 will do", "h1"))
    st.append(box([P("<b>Idea in one line:</b> use the true camera motion to un-bend the rolling-shutter photos, "
                     "then run the models again. If the extra error disappears, the bending was the cause.",
                     "box")], BOX_BLUE, BLUE))
    st.append(Spacer(1, 6))
    st += numbered([
        "Take each rolling-shutter photo from the same 1,124 clips.",
        "Use the gyroscope's measured turning to slide every row back to where it would be if the whole photo "
        "had been taken at once (“straightening”). This is a perfect-knowledge fix: the best case.",
        "Run the same three models on the straightened photos.",
        "Compare three versions of each clip: <b>normal camera</b>, <b>bent photos</b> and <b>straightened "
        "photos</b>.",
    ])
    st.append(Spacer(1, 4))
    st.append(table([
        ["If straightening removes…", "It means", "Decision"],
        ["half of the extra error or more", "The bending is the cause, and fixing it pays off.",
         "<font color='#1a7f37'><b>GO</b></font>: build our method"],
        ["a quarter to a half", "Partly the cause.", "Probably a workshop paper"],
        ["less than a quarter", "The gap comes from something else.",
         "<font color='#c4321b'><b>STOP</b></font> as a main paper"],
    ], [0.3 * W, 0.42 * W, 0.28 * W]))
    st.append(Spacer(1, 6))
    st += bullets([
        "<b>It also answers EXP0's open question.</b> If straightening fixes the slow clips too, the slow-clip "
        "gap was bending all along. If it does not, something else about camera 2 is involved.",
        "<b>One caution.</b> Straightening that uses only turning cannot undo the bend caused by walking. This "
        "rig walks fast (0.6–1.1 m per second), so we suggest a second version that also uses the walking "
        "motion. Otherwise, treat the turning-only result as a minimum.",
        "<b>Cost:</b> about 2–3 hours on the same GPU. It reuses EXP0's clips and scoring, so the numbers "
        "compare directly.",
    ])

    st.append(P("10. Our recommendations", "h1"))
    st += bullets([
        "<b>Run EXP1 next</b>, including the slow clips and a version that corrects for walking.",
        "<b>Make π³ our main model</b>, with VGGT second. π³ shows the biggest gap and it grows with speed. "
        "That makes it the best showcase for a fix.",
        "<b>Do not drop MapAnything blindly.</b> First check that our MapAnything settings match the other "
        "team's, since it is 2–6× less accurate than the others even on normal photos.",
        "<b>Housekeeping:</b> one idle, stuck download is left on the GPU box. Close it with "
        "<font face='Courier'>tmux kill-window -t rs3d:setup</font>.",
    ])

    # ------------------------------------------------------------------ appendix
    st.append(PageBreak())
    st.append(P("Appendix A. Plain-English glossary", "h1"))
    st.append(table([
        ["Term", "Meaning"],
        ["Rolling shutter", "Camera reads the photo row by row, top to bottom (here over 30 ms)."],
        ["Global shutter (normal)", "Camera captures every row at the same instant."],
        ["Clip (window)", "16 photos from about 2 seconds of a recording."],
        ["Path error (ATE)", "How far the model's guessed camera positions are from the true ones, on average "
                             "(cm), after lining up and scaling the guess."],
        ["Turning error (RPE-rot)", "How wrong the model is about how much the camera turned between neighbouring "
                                    "photos (degrees)."],
        ["Times worse (ratio)", "Rolling-shutter error ÷ normal error, using the middle value (median) of each "
                                "group of clips."],
        ["95% range (confidence interval)", "We resampled the clips 1,000 times. The true ratio lies in this range "
                                            "with 95% confidence."],
        ["Turning speed", "From the gyroscope. 1 radian per second ≈ 57° per second."],
        ["GO / GRAY / STOP", "The decision rules we agreed before running the experiment."],
    ], [0.3 * W, 0.7 * W]))

    st.append(P("Appendix B. Turning error, times worse (supporting score)", "h1"))
    st.append(table([
        ["Model", "Slow", "Medium", "Fast"],
        ["MapAnything", "1.01×", "1.10×", "1.26×"],
        ["VGGT", "1.47×", "1.54×", "1.68×"],
        ["π³", "1.30×", "1.61×", "1.80×"],
    ], [0.34 * W, 0.22 * W, 0.22 * W, 0.22 * W]))
    st.append(P("For turning error, all three models get steadily worse as speed rises.", "small"))

    st.append(P("Appendix C. Run facts", "h1"))
    st += bullets([
        "<b>Time:</b> about 1 hour 50 minutes from start to finish (setup 25 min, preparing photos 5 min, running "
        "the models about 70 min, analysis 5 min).",
        "<b>GPU:</b> rented A100 with 80 GB. At most 38.7 GB was in use. Cost about US$2.50 at $1.38 per hour. "
        "The other team's experiment on the same machine was not touched.",
        "<b>Exact model versions:</b> MapAnything (Apache-licence weights), VGGT-1B and π³. Code versions are "
        "recorded in <font face='Courier'>DATA_NOTES.md</font>.",
        "<b>Changes from the plan:</b> π³'s code has moved to a new web address; the three models ran side by "
        "side to save time; we added the fairness check and an extra finer speed breakdown.",
        "<b>All files</b> (results tables, technical charts and the detailed technical summary) are in "
        "<font face='Courier'>results/exp0/</font> on the GPU box (<font face='Courier'>~/rs3d/</font>) and in our "
        "project folder. The detailed version is <font face='Courier'>summary.md</font>.",
    ])

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="EXP0: Do 3D AI models break on rolling-shutter cameras?",
                            author="RS-3D team", subject="EXP0 gap test results")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
