"""OpenReview API v2 full-text search for RS terms; prints forum-level submissions with venue info.
Usage: python -I openreview_search.py out.json"""
import urllib.request, urllib.parse, json, sys, time, re
sys.stdout.reconfigure(encoding='utf-8')
TERMS = ['"rolling shutter"', '"rolling-shutter"', 'rolling shutter', 'jello', 'readout pose', 'rolling shutter VGGT', 'rolling shutter DUSt3R', 'rolling shutter feed-forward']
seen = {}
for term in TERMS:
    for off in range(0, 1000, 100):
        url = 'https://api2.openreview.net/notes/search?' + urllib.parse.urlencode({'term': term, 'type': 'terms', 'content': 'all', 'group': 'all', 'source': 'forum', 'limit': 100, 'offset': off})
        try:
            r = json.loads(urllib.request.urlopen(url, timeout=120).read())
        except Exception as e:
            print('ERR', term, off, e); time.sleep(10); break
        notes = r.get('notes', [])
        for n in notes:
            c = n.get('content', {})
            g = lambda k: (c.get(k, {}) or {}).get('value') if isinstance(c.get(k), dict) else c.get(k)
            title = g('title') or ''
            seen.setdefault(n.get('forum'), {'title': title, 'venue': g('venue'), 'venueid': g('venueid'), 'abstract': (g('abstract') or '')[:800],
                                              'cdate': n.get('cdate'), 'terms': set()})['terms'].add(term)
        print(term, off, len(notes), 'total?', r.get('count'))
        if len(notes) < 100: break
        time.sleep(2)
    time.sleep(2)
for k in seen: seen[k]['terms'] = sorted(seen[k]['terms'])
json.dump(seen, open(sys.argv[1], 'w'), indent=1)
pat = re.compile(r'rolling[\s\-]*shutter|jello', re.I)
print('unique forums', len(seen))
for k, v in sorted(seen.items(), key=lambda kv: -(kv[1]['cdate'] or 0)):
    if pat.search(v['title'] + ' ' + v['abstract']):
        print(time.strftime('%Y-%m-%d', time.gmtime((v['cdate'] or 0) / 1000)), '|', v['venue'], '|', v['venueid'], '|', v['title'][:120], '| https://openreview.net/forum?id=' + k)
