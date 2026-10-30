"""Fetch S2 citationCount/IDs for the 5 base papers (for coverage check). Writes s2_meta.json"""
import urllib.request, json, time, sys
sys.stdout.reconfigure(encoding='utf-8')
out = {}
for n, pid in [('VGGT','arXiv:2503.11651'),('DUSt3R','arXiv:2312.14132'),('MASt3R','arXiv:2406.09756'),('Pi3','arXiv:2507.13347'),('MapAnything','arXiv:2509.13414')]:
    for i in range(300):
        try:
            r = json.loads(urllib.request.urlopen(f'https://api.semanticscholar.org/graph/v1/paper/{pid}?fields=title,year,venue,citationCount,externalIds', timeout=60).read()); break
        except Exception as e:
            time.sleep(4)
    out[n] = r; print(n, r, flush=True); time.sleep(2)
json.dump(out, open('s2_meta.json', 'w'), indent=1)
