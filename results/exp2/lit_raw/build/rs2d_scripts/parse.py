import sys, xml.etree.ElementTree as ET
sys.stdout.reconfigure(encoding="utf-8")
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
for f in sys.argv[1:]:
    t=ET.parse(f); r=t.getroot()
    tot=r.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
    print('#####',f,'total',tot.text if tot is not None else '?')
    for e in r.findall('a:entry',ns):
        i=e.find('a:id',ns).text.split('/abs/')[-1]
        ti=' '.join(e.find('a:title',ns).text.split())
        pub=e.find('a:published',ns).text[:10]
        au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)]
        c=e.find('x:comment',ns); c=' '.join(c.text.split()) if c is not None else ''
        j=e.find('x:journal_ref',ns); j=' '.join(j.text.split()) if j is not None else ''
        d=e.find('x:doi',ns); d=d.text if d is not None else ''
        print(f"{i}|{pub}|{ti}|{'; '.join(au[:6])}{' et al.' if len(au)>6 else ''}|C:{c}|J:{j}|DOI:{d}")
