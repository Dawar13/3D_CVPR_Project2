"""Supplementary citation scan via OpenAlex (cites: filter, cursor pagination). Writes oa_<name>.json + prints summary."""
import urllib.request, urllib.parse, json, time, sys, re
sys.stdout.reconfigure(encoding='utf-8')
T = {'VGGT': 'W4413146238', 'DUSt3R': 'W4402816534', 'MASt3R': 'W4404002617', 'Pi3': 'W4416063870', 'MapAnything': 'W7162498741'}
pat = re.compile(r'rolling[\s\-]*shutter|jello|read-?out', re.I)
def abstract(inv):
    if not inv: return ''
    pos = {}
    for w, ps in inv.items():
        for p in ps: pos[p] = w
    return ' '.join(pos[i] for i in sorted(pos))
summ = {}
for name, wid in T.items():
    cursor, rows = '*', []
    while cursor:
        url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': f'cites:{wid}', 'per-page': 200, 'cursor': cursor, 'select': 'id,display_name,publication_year,abstract_inverted_index,ids'})
        for i in range(6):
            try:
                d = json.loads(urllib.request.urlopen(url, timeout=120).read()); break
            except Exception as e:
                print('err', e, file=sys.stderr); time.sleep(10)
        else:
            break
        res = d.get('results', [])
        rows += res
        cursor = d.get('meta', {}).get('next_cursor') if res else None
        time.sleep(0.5)
    hits = []
    for w in rows:
        a = abstract(w.get('abstract_inverted_index'))
        t = (w.get('display_name') or '') + ' ' + a
        m = sorted(set(x.lower() for x in pat.findall(t)))
        if m: hits.append({'id': w['id'], 'title': w.get('display_name'), 'year': w.get('publication_year'), 'match': m, 'ids': w.get('ids'), 'abstract': a[:1200]})
    summ[name] = {'openalex_id': wid, 'citing_scanned': len(rows), 'with_abstract': sum(1 for w in rows if w.get('abstract_inverted_index')), 'hits': len(hits)}
    json.dump({'hits': hits, 'n': len(rows)}, open(f'oa_{name}.json', 'w'), indent=0)
    print(name, summ[name], flush=True)
    for h in hits: print('   ', h['year'], h['match'], h['title'][:110])
json.dump(summ, open('oa_summary.json', 'w'), indent=1)
