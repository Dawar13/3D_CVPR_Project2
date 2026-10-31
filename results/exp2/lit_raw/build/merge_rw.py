"""Merge the four raw paper CSVs into results/exp2/related_work.csv (de-duplicated by normalised title).

Usage: python merge_rw.py <repo_root> [--report]
Writes <repo>/results/exp2/related_work.csv and a pickle of merged records (with primary topic) to the scratchpad.
"""
import csv, json, re, sys, unicodedata, os

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "results/exp2/lit_raw")
OUT = os.path.join(ROOT, "results/exp2/related_work.csv")
SCR = os.path.dirname(os.path.abspath(__file__))
SOURCES = [("scoop", "scoop.csv"), ("genre_models", "genre_models.csv"),
           ("rs2d_nerf_blur", "rs2d_nerf_blur.csv"), ("classical_slam_gyro", "classical_slam_gyro.csv")]
THREAT = {"low": 0, "med": 1, "high": 2}


def norm(t):
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t.strip())
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", t)


def topic_of(src, raw):
    raw = raw.strip()
    if src == "scoop":
        if "RS-geometry" in raw: return 4
        if "RS-per-scene-recon" in raw or "RS-data-synthesis" in raw: return 6
        if "RS-2D-correction" in raw: return 3
        if "RS-learned-classic" in raw or "adjacent-intra-frame-motion" in raw or "adjacent-RS-velocity-learned" in raw: return 7
        return 1
    if src == "genre_models":
        return 2 if raw.startswith("2") else 12
    if src == "rs2d_nerf_blur":
        if raw.startswith("T3-adjacent"): return 4
        if raw.startswith("T3/T9") or raw.startswith("T9"): return 9
        if raw.startswith("T6"): return 6
        return 3
    # classical
    return int(raw.split(",")[0])


rows = []
for src, fn in SOURCES:
    with open(os.path.join(RAW, fn), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r = {k: (v or "").strip() for k, v in r.items()}
            r["_src"] = src
            r["_topic"] = topic_of(src, r["topic"])
            rows.append(r)

groups = {}
order = []
for r in rows:
    k = norm(r["title"])
    if k not in groups:
        groups[k] = []
        order.append(k)
    groups[k].append(r)

# ---- explicit resolutions of cross-slice conflicts (each one is listed in LIT_REVIEW.md) ----
VENUE_OVERRIDE = {
    # classical slice opened the PMLR page; scoop/rs2d saw only arXiv
    norm("IMU-Assisted Learning of Single-View Rolling Shutter Correction"):
        ("CoRL 2021 = PMLR vol 164 (2022) pp 861-870 (PMLR page; classical slice). arXiv v1 2020; scoop/rs2d slices saw no venue on the arXiv page", "2022"),
    # rs2d slice opened the AAAI OJS page; classical slice did not
    norm("Single Image Rolling Shutter Removal with Diffusion Models"):
        ("AAAI 2025 (AAAI OJS page: Vol.39 No.9 pp.9373-9381; rs2d slice). Classical slice left it as arXiv 2024 because it did not open the AAAI page", "2025"),
    # scoop slice saw it in the CVF CVPR 2025 listing
    norm("Order-One Rolling Shutter Cameras"):
        ("CVPR 2025 (CVF CVPR 2025 listing; scoop slice). Classical slice: arXiv 2024, no acceptance stated on the arXiv page", "2025"),
    norm("Rolling Shutter Relative Pose Estimation Made Practical"):
        ("ECCV 2026 (ECCV 2026 accepted-papers list and repo README per scoop slice; Google Research publication page per classical slice; arXiv comments field empty)", "2026"),
    norm("RnGCam: High-speed video from rolling & global shutter measurements"):
        ("arXiv (rs2d slice: no venue on arXiv page). The scoop slice lists RnGCam among ICCV 2025 CVF titles; venue UNVERIFIED until a proceedings page is opened", "2025"),
}
ID_OVERRIDE = {
    norm("Rolling Shutter Image as an IMU: Estimating Camera Motion from a Single Rolling Shutter Image"):
        "none (course report; no arXiv ID or DOI found)",
    norm("Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion"):
        "none found (no arXiv ID or venue listed on the project page or PDF)",
}
TOPIC_OVERRIDE = {
    norm("Learning Structure-And-Motion-Aware Rolling Shutter Correction"): 7,
    norm("IMU-Assisted Learning of Single-View Rolling Shutter Correction"): 7,
    norm("Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion"): 1,
    norm("Continuous-Time Spatiotemporal Calibration of a Rolling Shutter Camera-IMU System"): 5,
}
TOPIC_PRIORITY = ["rs2d_nerf_blur", "classical_slam_gyro", "genre_models", "scoop"]

ARX = re.compile(r"\d{4}\.\d{4,5}|cs/\d{7}")
DOI = re.compile(r"10\.\d{4,9}/[^\s;,)]+")


def longest(vals):
    vals = [v for v in vals if v]
    return max(vals, key=len) if vals else ""


merged = []
conflicts = []
for k in order:
    g = groups[k]
    srcs = [r["_src"] for r in g]
    rec = {}
    rec["title"] = g[0]["title"]
    rec["authors"] = longest([r["authors"].replace("; ", ", ") for r in g])
    if k in VENUE_OVERRIDE:
        rec["venue"], rec["year"] = VENUE_OVERRIDE[k]
    else:
        rec["venue"] = longest([r["venue"] for r in g])
        years = sorted(set(r["year"] for r in g if r["year"]))
        rec["year"] = [r["year"] for r in g if r["venue"] == rec["venue"]][0] if rec["venue"] else (years[0] if years else "")
        if len(years) > 1:
            conflicts.append(f"year differs for '{rec['title']}': {years}")
    if k in ID_OVERRIDE:
        rec["id"] = ID_OVERRIDE[k]
    else:
        base = longest([r["id"] for r in g])
        extra = []
        for r in g:
            for m in ARX.findall(r["id"]) + DOI.findall(r["id"]):
                if m.lower() not in base.lower() and m.lower() not in " ".join(extra).lower():
                    extra.append(("arXiv:" + m) if ARX.fullmatch(m) else ("DOI " + m))
        rec["id"] = "; ".join([base] + extra) if base else "; ".join(extra)
    rec["what_it_does"] = longest([r["what_it_does"] for r in g])
    rec["what_it_does_not_do"] = longest([r["what_it_does_not_do"] for r in g])
    codes = [(r["_src"], r["code"]) for r in g if r["code"]]
    ycodes = [c for c in codes if c[1].startswith("Y")]
    if ycodes:
        chosen_src, chosen = max(ycodes, key=lambda c: len(c[1]))
        notes = [f"{s} slice recorded: {c[:90]}{'...' if len(c) > 90 else ''}" for s, c in codes if not c.startswith("Y")]
        if notes:
            chosen += " [cross-slice note: " + " | ".join(notes) + "]"
            conflicts.append(f"code differs for '{rec['title']}': chose {chosen_src}")
        rec["code_available"] = chosen
    else:
        checked = [c for _, c in codes if "not checked" not in c]
        rec["code_available"] = longest(checked) or longest([c for _, c in codes])
    rec["how_we_differ"] = longest([r["how_we_differ"] for r in g])
    th = [r["threat"].lower() for r in g if r["threat"].lower() in THREAT]
    rec["threat_level"] = max(th, key=lambda t: THREAT[t]) if th else ""
    if len(set(th)) > 1:
        conflicts.append(f"threat differs for '{rec['title']}': {[(r['_src'], r['threat']) for r in g]} -> {rec['threat_level']}")
    venues = set(r["venue"] for r in g)
    if len(venues) > 1:
        conflicts.append(f"venue strings differ for '{rec['title']}': {[(r['_src'], r['venue']) for r in g]}")
    for f in ["title", "authors", "venue", "year", "id", "what_it_does", "what_it_does_not_do", "code_available", "how_we_differ", "threat_level"]:
        if not rec[f]:
            rec[f] = "UNVERIFIED"
    # primary topic
    if k in TOPIC_OVERRIDE:
        rec["_topic"] = TOPIC_OVERRIDE[k]
    else:
        for s in TOPIC_PRIORITY:
            hit = [r for r in g if r["_src"] == s]
            if hit:
                rec["_topic"] = hit[0]["_topic"]
                break
    rec["_srcs"] = srcs
    rec["_verified"] = " | ".join(f"{r['_src']}: {r['verified']}" for r in g)
    merged.append(rec)

# order: threat (med first), then primary topic, then year desc
merged.sort(key=lambda r: (-THREAT.get(r["threat_level"], -1), r["_topic"], -int(re.sub(r"\D", "", r["year"])[:4] or 0)))

COLS = ["title", "authors", "venue", "year", "id", "what_it_does", "what_it_does_not_do", "code_available", "how_we_differ", "threat_level"]
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for r in merged:
        w.writerow([r[c] for c in COLS])

with open(os.path.join(SCR, "merged.json"), "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=0)

if "--report" in sys.argv:
    sys.stdout.reconfigure(encoding="utf-8")
    print("raw rows:", len(rows), " merged:", len(merged))
    for k in order:
        if len(groups[k]) > 1:
            print("DUP", [r["_src"] for r in groups[k]], "|", groups[k][0]["title"][:80])
    print("--- conflicts")
    for c in conflicts:
        print(c)
    from collections import Counter
    print(Counter(r["_topic"] for r in merged))
    print(Counter(r["threat_level"] for r in merged))
