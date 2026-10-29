"""Grep CVF open-access 'day=all' listing HTML for rolling-shutter-related titles. Usage: python -I cvf_grep.py file.html [...]"""
import re, sys, html
sys.stdout.reconfigure(encoding='utf-8')
pat = re.compile(r'rolling|shutter|readout|read-out|scanline|scan-line|jello|wobble|row-wise|exposure', re.I)
for f in sys.argv[1:]:
    t = open(f, encoding='utf-8', errors='replace').read()
    titles = re.findall(r'<dt class="ptitle"><br><a href="([^"]+)">(.*?)</a>', t, re.S)
    hits = [(h, html.unescape(x)) for h, x in titles if pat.search(x)]
    print('###', f, 'titles:', len(titles), 'hits:', len(hits))
    for h, x in hits:
        print('   ', x, '|', h)
