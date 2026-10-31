"""Curate results/exp2/lit_raw/datasets.csv into results/exp2/datasets.csv (adds our status and frame-to-GT timing).

Usage: python curate_ds.py <repo_root>
"""
import csv, os, sys

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "results/exp2/lit_raw/datasets.csv")
OUT = os.path.join(ROOT, "results/exp2/datasets.csv")

rows = list(csv.DictReader(open(RAW, encoding="utf-8", newline="")))
assert len(rows) == 36, len(rows)

NL = "not in the 2026-10-29 download list"
# our_status by raw row index (1-based). VM facts are measured on 2026-10-29 (reported by the main agent).
STATUS = {
    1: "IN HAND, PREPPED (measured on VM 2026-10-29): re-downloaded 10 sequences + dataset-calib-imu1 (GS-GS control), ~22 GB extracted; prep gives 1124 kept windows, identical to EXP0",
    2: "IN HAND (measured on VM 2026-10-29): 23 sequences downloaded. CC BY-NC 4.0 approved by the user for research use (2026-10-29)",
    3: "NOT AVAILABLE: official URL returned HTTP 404 (2026-10-29); not planned",
    4: f"LEAD ONLY: licence and RS statement need a human check; not requested; {NL}",
    5: f"LEAD ONLY: Baidu Netdisk, GT method UNVERIFIED; not requested; {NL}",
    6: f"NOT PLANNED: GT ~10 cm, too coarse for RS-sized errors; {NL}",
    7: "WAITING on user sign-up (HUMAN_TODO.md item 2)",
    8: f"NOT REQUESTED (registration form); {NL}",
    9: f"OPTIONAL: source of real phone readout values only (no GT); no sign-up; {NL}",
    10: "NOT FOUND: no download located",
    11: "WAITING on user sign-up (HUMAN_TODO.md item 1)",
    12: "WAITING on user application (HUMAN_TODO.md item 3)",
    13: f"NOT REQUESTED: ScanNet++ covers the same need (HUMAN_TODO.md 'Not needed'); {NL}",
    14: "IN HAND (measured on VM 2026-10-29): 18 scenes downloaded. mesh.ply = binary PLY with per-vertex xyz, normals and uchar RGB colours, and polygon (quad) faces; e.g. room_0 has 954,492 vertices and 953,647 faces. Use approved by the user under the Replica Dataset Research Terms (2026-10-29)",
    15: f"NOT REQUESTED: Replica covers the mesh need (HUMAN_TODO.md 'Not needed'); {NL}",
    16: f"NOT REQUESTED: Replica covers the mesh need (HUMAN_TODO.md 'Not needed'); {NL}",
    17: f"NOT REQUESTED: Replica covers the mesh need (HUMAN_TODO.md 'Not needed'); {NL}",
    18: f"NOT PLANNED: no meshes; TartanAir V2 preferred; {NL}",
    19: f"APPROVED by the user for research use (CC BY 4.0, 2026-10-29); {NL}",
    20: f"NOT PLANNED: RS simulation not documented, no meshes; {NL}",
    21: f"LICENCE APPROVED by the user (ETH3D, CC BY-NC-SA 4.0, 2026-10-29); MVS part {NL} (only the SLAM training set was downloaded)",
    22: "IN HAND (measured on VM 2026-10-29): training set downloaded, 61 sequences, mono + IMU. CC BY-NC-SA 4.0 approved by the user (2026-10-29). 61 matches the raw audit's training split (56 mocap + 5 SfM-GT training sequences)",
    23: f"CONDITIONAL: use only after a human confirms GS from the IJRR paper; {NL}",
    24: f"CANDIDATE: GS driving counterpart to Waymo; terms-by-use, so a human must decide; {NL}",
    25: f"NOT PLANNED as a GS control (shutter UNVERIFIED); {NL}",
    26: "DO NOT COUNT AS GS CONTROL (phone video; COLMAP GT absorbs RS); not planned",
    27: "DO NOT COUNT AS GS CONTROL (consumer/YouTube video; SLAM+BA GT on the same frames); not planned",
    28: "DO NOT COUNT AS GS CONTROL (shutter unknown); terms-by-use; not planned",
    29: f"CANDIDATE: rendered GS benchmark and small CC-BY/CC-0 mesh source; {NL}",
    30: f"CANDIDATE: GS no-harm set (static capture); {NL}",
    31: "DO NOT COUNT AS GS CONTROL (iPad RS presumed); signed ToU needed; not requested",
    32: f"NOT PLANNED (licence UNVERIFIED); {NL}",
    33: "DO NOT COUNT AS GS CONTROL (shutter unknown); not planned",
    34: "DO NOT COUNT AS GS CONTROL (Kinect shutter unknown); not planned",
    35: f"CANDIDATE for 2D-unroller sanity checks / fine-tuning only (no poses); {NL}",
    36: f"2D-unroller training sets only; the original Google Drive links return HTTP 404 per the rs2d slice; {NL}",
}

# frame-to-GT timing we use (or the raw audit's statement + what remains to verify)
TV = "to verify from files (main agent)"
TIMING = {
    1: "OURS: RS GT at the centre-row time = first-row stamp + 511.5 x 29.4737 us = +15.08 ms; sensitivity run with GT at the first-row stamp (as in EXP0). GS cam0: GT at the stamp (middle of exposure)",
    2: f"Raw audit: sensors synced, iPhone stamps converted to time since device boot; exposure reference row/point not stated; readout and scan direction unknown (video stored 1280x720 portrait). Which row the stamp refers to and the sensor row direction in the stored frame: {TV}",
    7: "Docs: t(p) = t_image + (row(p)/H - 0.5) x readout, so the image stamp is the middle row; GT at the stamp. Sensor-row orientation vs stored image must be checked",
    11: "Devkit: t(row) = camera_trigger_time + 0.5 x shutter + readout x row/H (TOP_TO_BOTTOM; other directions per enum); proto 'time image center = middle of exposure of middle row'. pose is the vehicle pose at pose_timestamp; camera pose needs the extrinsic",
    12: "Not documented; readout would have to be estimated; pseudo-GT is one pose per frame (biased by RS)",
    14: "OURS (synthetic, src/rsgen): frame time = centre-line time; the GS twin is rendered at the same centre-line instant; readout and scan direction chosen by us (tb/bt/lr validated in results/exp2/generator/validation_testroom.json)",
    19: "OURS (synthetic): RS by row-stitching the 1000 Hz lcam_front video; reference row chosen by us; how poses are given for the 1000 Hz stream is UNVERIFIED",
    22: f"GS rig, so no row timing. Raw audit: convention not stated; camera/IMU ns stamps converted to mocap time via sequence_calibration.txt. Exposure reference point (start vs middle): {TV}",
}

COLS = ["name", "category", "shutter", "readout_ms", "scan_direction", "gt_poses", "gt_depth", "gt_3d_scan", "imu",
        "gs_twin", "indoor_outdoor", "dynamic_objects", "size", "licence", "download_method", "needs_human",
        "frame_to_gt_timing", "our_status", "verified", "notes"]

out = []
for i, r in enumerate(rows, 1):
    o = {
        "name": r["name"], "category": r["category"], "shutter": r["shutter"], "readout_ms": r["readout_ms"],
        "scan_direction": r["scan_direction"], "gt_poses": r["gt_poses"], "gt_depth": r["gt_depth"],
        "gt_3d_scan": r["gt_3d"], "imu": r["imu"], "gs_twin": r["gs_twin"], "indoor_outdoor": r["indoor_outdoor"],
        "dynamic_objects": r["dynamic"], "size": r["size"], "licence": r["license"], "download_method": r["download"],
        "needs_human": r["needs_human"], "frame_to_gt_timing": TIMING.get(i, r["timestamp_convention"]),
        "our_status": STATUS[i], "verified": r["verified"], "notes": r["notes"],
    }
    if i == 1:
        o["size"] += "; measured on our VM: ~22 GB extracted for 10 sequences + dataset-calib-imu1"
    if i == 14:
        o["needs_human"] += " -> approved by the user 2026-10-29"
    if i in (2, 19, 21, 22):
        o["needs_human"] += " (licence approved by the user 2026-10-29)"
    if i == 35:  # BS-RSC: fill from the rs2d slice (paper values), attributed
        o["sensor_note"] = ""
        o["notes"] += (" | rs2d slice (from the CVPR'22 paper): RS FLIR FL3-U3-13S2C (CMOS) + GS FLIR GS3-U3-28S4C (CCD), "
                       "1024x768, 25 fps, 1 ms exposure (no blur); GS aligned to the RS middle scanline; 80 videos x 50 frames (50/15/15); "
                       "readout value not stated; GDrive BS-RSC.zip resolves (2026-10-29)")
        o["size"] = "80 videos x 50 frames (rs2d slice, from paper); GB unknown"
    if i == 36:
        o["download_method"] = ("README Google Drive links (datasets slice). CONFLICT: the rs2d slice re-checked all four archives on "
                                "2026-10-29 and they return HTTP 404. Fastec-RS mirrors: JCD repo fastec_rs_lmdb.zip (GDrive, resolves) and "
                                "HF yunfanlu/UNIINR-Fastec-Simulated-Dataset (68 GB, contents not inspected). No working Carla-RS mirror found")
        o["readout_ms"] += "; gamma (readout ratio) = 1 per RSSR-TPAMI (rs2d slice)"
        o["gt_depth"] += "; Carla-RS loader reads per-pixel RS depth, flow, occlusion mask and GT velocity (rs2d slice)"
    o.pop("sensor_note", None)
    for c in COLS:
        if not o.get(c):
            o[c] = "UNVERIFIED"
    out.append(o)

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(out)
print("wrote", OUT, len(out), "rows")
