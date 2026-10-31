"""Fill {{DSTABLES}} in the data-audit template from results/exp2/datasets.csv. Usage: python fill_audit.py <repo> <tpl> <out>"""
import csv, os, re, sys
repo, tpl, out = sys.argv[1:4]
rows = list(csv.DictReader(open(os.path.join(repo, "results/exp2/datasets.csv"), encoding="utf-8", newline="")))
BAR = chr(92) + "|"

def cut(s, n=90):
    """Keep the head of a field up to the first ' - ' or ' (' (used for YES/NO-style fields)."""
    s = re.split(r" - | \(", s, maxsplit=1)[0].strip()
    return trunc(s, n)

def trunc(s, n):
    return (s[:n].rstrip() + "...") if len(s) > n else s

def e(s):
    return s.replace("|", BAR)

def status_short(s):
    if ":" in s[:60]:
        return s.split(":")[0]
    return s if len(s) <= 120 else s.split("; ")[0]

a = ["**Table A. Sensor and ground truth**", "",
     "| # | Dataset | Shutter | Readout (ms) | Scan direction | GT poses | GT depth | GT 3D / scan | IMU | GS twin | Indoor / outdoor | Dynamic |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
b = ["**Table B. Size, licence, access and our status**", "",
     "| # | Dataset | Size | Licence | Download | needs_human | Our status |",
     "|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    a.append("| " + " | ".join([str(i), e(r["name"]), e(cut(r["shutter"], 70)), e(trunc(r["readout_ms"], 70)), e(trunc(r["scan_direction"], 70)),
              e(cut(r["gt_poses"])), e(cut(r["gt_depth"])), e(cut(r["gt_3d_scan"])), e(cut(r["imu"])), e(cut(r["gs_twin"])),
              e(cut(r["indoor_outdoor"])), e(cut(r["dynamic_objects"]))]) + " |")
    b.append("| " + " | ".join([str(i), e(r["name"]), e(trunc(r["size"], 140)), e(r["licence"]), e(trunc(r["download_method"], 110)),
              e(r["needs_human"]), e(status_short(r["our_status"]))]) + " |")
t = open(tpl, encoding="utf-8").read().replace("{{DSTABLES}}", "\n".join(a) + "\n\n" + "\n".join(b))
t = t.replace('Long fields are cut at the first " - " or " ("; the CSV has the full text.',
              'Yes/no-style fields are cut at the first " - " or " ("; long fields are truncated with "...". The CSV has the full text.')
assert "{{" not in t
open(out, "w", encoding="utf-8", newline="\n").write(t)
print("wrote", out, t.count("\n") + 1, "lines")
