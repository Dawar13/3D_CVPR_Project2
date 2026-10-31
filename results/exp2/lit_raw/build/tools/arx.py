import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time
sys.stdout.reconfigure(encoding="utf-8")
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
def q(url):
    for i in range(3):
        try:
            d=urllib.request.urlopen(url,timeout=60).read()
            if d: return d
        except Exception as e:
            print('ERR',e); time.sleep(3)
    return b''
mode=sys.argv[1]; arg=sys.argv[2]
full = len(sys.argv)>3 and sys.argv[3]=='full'
if mode=='id':
    url='https://export.arxiv.org/api/query?id_list='+arg+'&max_results=50'
else:
    url='https://export.arxiv.org/api/query?search_query='+arg+'&sortBy=submittedDate&sortOrder=descending&max_results='+(sys.argv[4] if len(sys.argv)>4 else '50')
d=q(url)
r=ET.fromstring(d)
for e in r.findall('a:entry',ns):
    i=e.find('a:id',ns).text.split('/abs/')[-1]
    t=' '.join(e.find('a:title',ns).text.split())
    au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)]
    c=e.find('x:comment',ns); c=c.text if c is not None else ''
    jr=e.find('x:journal_ref',ns); jr=jr.text if jr is not None else ''
    p=e.find('a:published',ns).text[:10]
    print(f"{i} | {p} | {t} | {', '.join(au[:6])}{' et al.' if len(au)>6 else ''} | C: {' '.join(c.split())} {('| JR: '+jr) if jr else ''}")
    if full: print('   ABS:', ' '.join(e.find('a:summary',ns).text.split())); print()
