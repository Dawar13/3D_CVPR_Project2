"""Grep downloaded arXiv HTML full texts for RS / intra-frame-time terms. Usage: python -I html_grep.py file1.html ..."""
import re, sys, html
sys.stdout.reconfigure(encoding='utf-8')
pat = re.compile(r'rolling[\s\-]*shutter|readout|read-out|scanline|exposure time|intra-frame|during exposure|feed-forward|feedforward|velocit', re.I)
for f in sys.argv[1:]:
    t = open(f, encoding='utf-8', errors='replace').read()
    title = re.search(r'<title>(.*?)</title>', t, re.S)
    txt = html.unescape(re.sub(r'<[^>]+>', ' ', t)); txt = re.sub(r'\s+', ' ', txt)
    print('=====', f, '|', title.group(1).strip()[:120] if title else '', '| chars', len(txt))
    counts = {}
    for m in pat.finditer(txt): counts[m.group(0).lower()] = counts.get(m.group(0).lower(), 0) + 1
    print('   counts:', counts)
    for m in re.finditer(r'rolling[\s\-]*shutter', txt, re.I):
        print('   RS ctx:', txt[max(0, m.start()-220):m.end()+220])
