"""arXiv API scan: every paper with 'rolling shutter' / 'rolling-shutter' in title or abstract,
submitted since 2025-01-01. Writes JSON + TSV next to this script."""
import urllib.request, urllib.parse, time, json, re, sys, os
import xml.etree.ElementTree as ET

OUT = os.path.dirname(os.path.abspath(__file__))
NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
QUERIES = [
    'abs:"rolling shutter"',
    'ti:"rolling shutter"',
    'abs:"rolling-shutter"',
    'ti:"rolling-shutter"',
    'abs:"rolling shutters"',
    'abs:rolling AND abs:shutter',
]
DATE = ' AND submittedDate:[202501010000 TO 202610302359]'
papers = {}
log = []
for q in QUERIES:
    start = 0
    total = None
    while True:
        params = {'search_query': q + DATE, 'sortBy': 'submittedDate', 'sortOrder': 'descending',
                  'start': start, 'max_results': 200}
        url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode(params)
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                break
            except Exception as e:
                print('retry', e, file=sys.stderr); time.sleep(5)
        root = ET.fromstring(data)
        tr = root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
        total = int(tr.text) if tr is not None else 0
        entries = root.findall('a:entry', NS)
        for e in entries:
            aid = e.find('a:id', NS).text.split('/abs/')[-1]
            base = re.sub(r'v\d+$', '', aid)
            title = ' '.join(e.find('a:title', NS).text.split())
            abstract = ' '.join(e.find('a:summary', NS).text.split())
            pub = e.find('a:published', NS).text
            upd = e.find('a:updated', NS).text
            authors = [a.find('a:name', NS).text for a in e.findall('a:author', NS)]
            com = e.find('arxiv:comment', NS)
            jr = e.find('arxiv:journal_ref', NS)
            cats = [c.get('term') for c in e.findall('a:category', NS)]
            rec = papers.setdefault(base, {'id': base, 'title': title, 'abstract': abstract, 'published': pub,
                                           'updated': upd, 'authors': authors,
                                           'comment': com.text if com is not None else '',
                                           'journal_ref': jr.text if jr is not None else '',
                                           'cats': cats, 'queries': []})
            rec['queries'].append(q)
        log.append((q + DATE, start, total, len(entries)))
        print(q, start, total, len(entries), file=sys.stderr)
        start += 200
        time.sleep(3.5)
        if start >= total or not entries:
            break

# strict filter: phrase in title or abstract
pat = re.compile(r'rolling[\s\-]+shutter', re.I)
strict = {k: v for k, v in papers.items() if pat.search(v['title']) or pat.search(v['abstract'])}
json.dump({'log': log, 'all': papers, 'strict_ids': sorted(strict)}, open(os.path.join(OUT, 'arxiv_scan.json'), 'w'), indent=1)
print('union', len(papers), 'strict', len(strict), file=sys.stderr)
