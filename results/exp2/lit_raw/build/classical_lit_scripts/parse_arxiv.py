import sys, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
for f in sys.argv[1:]:
    t=ET.parse(f).getroot()
    print("=== ",f, len(t.findall('a:entry',ns)))
    for e in t.findall('a:entry',ns):
        i=e.find('a:id',ns).text.split('/abs/')[-1]
        d=e.find('a:published',ns).text[:10]
        ti=' '.join(e.find('a:title',ns).text.split())
        au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)]
        c=e.find('x:comment',ns); c=' '.join(c.text.split()) if c is not None else ''
        j=e.find('x:journal_ref',ns); j=' '.join(j.text.split()) if j is not None else ''
        print(f"{i} | {d} | {ti} | {', '.join(au[:4])}{' et al.' if len(au)>4 else ''} | C:{c} | J:{j}")
