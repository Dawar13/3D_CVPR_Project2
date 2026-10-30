"""Semantic Scholar: resolve base-model papers, paginate ALL citing papers, filter for RS keywords.
Usage: python -I s2_cites.py   (writes s2_<name>.json and s2_summary.json)"""
import urllib.request, urllib.parse, json, time, sys, re, os
sys.stdout.reconfigure(encoding='utf-8')
BASE = 'https://api.semanticscholar.org/graph/v1'
def get(url, tries=400):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'rs3d-litreview/0.1'})
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            print('  HTTP', e.code, body, file=sys.stderr, flush=True)
            if e.code in (400, 404):
                return {'_error': e.code, '_body': body.decode('utf-8', 'replace')}
            time.sleep(4)
        except Exception as e:
            print('  ERR', repr(e), file=sys.stderr, flush=True); time.sleep(15)
    return {'_error': 'gave_up'}

TARGETS = {
    'VGGT': None if os.environ.get('SKIP_VGGT', '1') == '1' else 'arXiv:2503.11651',
    'DUSt3R': 'arXiv:2312.14132',
    'MASt3R': sys.argv[1] if len(sys.argv) > 1 else None,
    'Pi3': sys.argv[2] if len(sys.argv) > 2 else None,
    'MapAnything': sys.argv[3] if len(sys.argv) > 3 else None,
}
pat = re.compile(r'rolling[\s\-]*shutter|jello|read-?out', re.I)
summary = {}
for name, pid in TARGETS.items():
    if not pid: continue
    meta = {'note': 'meta lookup skipped in run 2 to save rate budget'}
    rows, off, err = [], 0, None
    while True:
        url = f'{BASE}/paper/{pid}/citations?' + urllib.parse.urlencode({'fields': 'title,abstract,externalIds,year,venue,publicationDate,authors', 'limit': 1000, 'offset': off})
        r = get(url)
        if '_error' in r:
            err = r; print(name, 'stop at offset', off, r, flush=True); break
        data = r.get('data', [])
        rows += [d['citingPaper'] for d in data]
        print(name, 'offset', off, 'got', len(data), 'next', r.get('next'), flush=True)
        if 'next' not in r or not data: break
        off = r['next']; time.sleep(3)
    hits = []
    for p in rows:
        t = (p.get('title') or '') + ' ' + (p.get('abstract') or '')
        m = sorted(set(x.lower() for x in pat.findall(t)))
        if m: hits.append({'title': p.get('title'), 'year': p.get('year'), 'venue': p.get('venue'), 'ids': p.get('externalIds'), 'match': m,
                           'abstract': (p.get('abstract') or '')[:1500], 'authors': [a.get('name') for a in (p.get('authors') or [])][:8]})
    n_abs = sum(1 for p in rows if p.get('abstract'))
    summary[name] = {'paper_id': pid, 'meta': meta, 'citing_scanned': len(rows), 'with_abstract': n_abs, 'hits': len(hits), 'stop_error': err}
    json.dump({'meta': meta, 'rows': rows, 'hits': hits}, open(f's2_{name}.json', 'w'), indent=0)
    old = json.load(open('s2_summary.json')) if os.path.exists('s2_summary.json') else {}
    old.update(summary); json.dump(old, open('s2_summary.json', 'w'), indent=1)
    print('DONE', name, summary[name]['citing_scanned'], 'abs', n_abs, 'hits', len(hits), flush=True)
