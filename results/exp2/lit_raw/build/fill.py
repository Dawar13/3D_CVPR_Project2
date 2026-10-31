"""Fill {{TABLE:n}} and {{CITE:substring}} placeholders in a markdown template from merged.json.

Usage: python fill.py <template.md> <out.md>
"""
import json, os, re, sys

SCR = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(SCR, "merged.json"), encoding="utf-8"))
THREAT = {"high": 0, "med": 1, "low": 2}


def esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


def code_short(c):
    if c.startswith("Y"):
        u = re.search(r"https?://[^\s;,)\]]+", c)
        lic = ""
        pats = [(r"\bMIT\b", "MIT"), (r"Apache-2\.0", "Apache-2.0"), (r"BSD-3", "BSD-3-Clause"), (r"GPL-3\.0", "GPL-3.0"),
                (r"CC BY-NC-SA 4\.0", "CC BY-NC-SA 4.0"), (r"CC BY-NC 4\.0", "CC BY-NC 4.0"),
                (r"research[- ](only|purpose)", "research-only"), (r"license: none|no license|no LICENSE", "no licence")]
        for pat, name in pats:
            if re.search(pat, c):
                lic = name
                break
        return "Y " + (u.group(0) if u else "") + (f" ({lic})" if lic else "")
    if c.startswith("UNVERIFIED"):
        return "UNVERIFIED"
    return "N"


def table(topic):
    rows = [r for r in M if r["_topic"] == topic]
    rows.sort(key=lambda r: (THREAT.get(r["threat_level"], 3), -int(re.sub(r"\D", "", r["year"])[:4] or 0), r["title"].lower()))
    out = [f"*Papers whose primary topic is {topic} ({len(rows)}; generated from `related_work.csv`, fields as in the raw files).*", "",
           "| # | Title | Authors | Venue (as recorded, with source) | Year | ID | Code | Threat |",
           "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        out.append(f"| {i} | {esc(r['title'])} | {esc(r['authors'])} | {esc(r['venue'])} | {esc(r['year'])} | {esc(r['id'])} | {esc(code_short(r['code_available']))} | {r['threat_level']} |")
    return "\n".join(out)


def cite(sub):
    hits = [r for r in M if sub.lower() in r["title"].lower()]
    if len(hits) != 1:
        raise SystemExit(f"CITE '{sub}' matched {len(hits)} records: {[h['title'] for h in hits]}")
    r = hits[0]
    a = r['authors'] if r['authors'].endswith('.') else r['authors'] + '.'
    return f"*{r['title']}*. {a} {r['venue']}. {r['year']}. {r['id']}"


tpl = open(sys.argv[1], encoding="utf-8").read()
tpl = re.sub(r"\{\{TABLE:(\d+)\}\}", lambda m: table(int(m.group(1))), tpl)
tpl = re.sub(r"\{\{CITE:([^}]+)\}\}", lambda m: cite(m.group(1)), tpl)
left = re.findall(r"\{\{[^}]*\}\}", tpl)
assert not left, left
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(tpl)
print("wrote", sys.argv[2], tpl.count("\n") + 1, "lines")
