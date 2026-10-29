"""Adversarial arXiv API queries combining RS / intra-frame-motion terms with feed-forward 3D model terms."""
import urllib.request, urllib.parse, time, json, re, sys, os
import xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding='utf-8')
NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
QS = [
 'all:shutter AND all:DUSt3R', 'all:shutter AND all:VGGT', 'all:shutter AND all:MASt3R', 'all:shutter AND all:pointmap',
 'all:shutter AND all:"feed-forward"', 'all:shutter AND all:"foundation model"', 'all:shutter AND all:CUT3R',
 'all:shutter AND all:MonST3R', 'all:shutter AND all:"Depth Anything"', 'all:shutter AND all:MapAnything',
 'all:jello', 'all:readout AND all:pointmap', 'all:readout AND all:VGGT', 'all:readout AND all:DUSt3R',
 'all:"motion blur" AND all:DUSt3R', 'all:"motion blur" AND all:VGGT', 'all:"motion blur" AND all:pointmap',
 'all:blur AND all:MASt3R', 'all:event AND all:VGGT', 'all:event AND all:DUSt3R', 'all:"event camera" AND all:pointmap',
 'all:velocity AND all:pointmap', 'all:velocity AND all:VGGT', 'all:"scene flow" AND all:VGGT',
 'all:"intra-frame" AND all:pose', 'all:"per-row" AND all:pose', 'all:"row-wise" AND all:pose AND all:camera',
 'all:"exposure" AND all:VGGT', 'all:"continuous-time" AND all:VGGT', 'all:"continuous-time" AND all:DUSt3R',
 'all:"rolling shutter" AND all:transformer', 'all:"rolling shutter" AND all:learning AND all:pose',
 'all:"rolling shutter" AND all:depth', 'all:"rolling shutter" AND all:reconstruction',
]
res = {}
for q in QS:
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'sortBy': 'submittedDate', 'sortOrder': 'descending', 'max_results': 100})
    root = None
    for att in range(8):
        try:
            data = urllib.request.urlopen(url, timeout=90).read(); root = ET.fromstring(data); break
        except Exception as e:
            print("  err", repr(e), file=sys.stderr); time.sleep(20 + 10*att)
    if root is None:
        print("##", q, "-> FAILED"); continue
    tot = root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text
    ents = []
    for e in root.findall('a:entry', NS):
        ents.append({'id': e.find('a:id', NS).text.split('/abs/')[-1], 'pub': e.find('a:published', NS).text[:10],
                     'title': ' '.join(e.find('a:title', NS).text.split()),
                     'abstract': ' '.join(e.find('a:summary', NS).text.split()),
                     'authors': [a.find('a:name', NS).text for a in e.findall('a:author', NS)],
                     'comment': (e.find('arxiv:comment', NS).text if e.find('arxiv:comment', NS) is not None else '')})
    res[q] = {'total': tot, 'entries': ents}
    print('##', q, '->', tot)
    for x in ents[:40]:
        print('   ', x['pub'], x['id'], x['title'][:110])
    time.sleep(6)
json.dump(res, open('arxiv_adv.json', 'w'), indent=1)
