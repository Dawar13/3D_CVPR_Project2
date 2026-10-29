import json,re,sys
sys.stdout.reconfigure(encoding='utf-8')
d=json.load(open(sys.argv[1]))
kw=re.compile(r'dust3r|mast3r|vggt|feed-?forward|foundation|pointmap|point map|transformer|cut3r|monst3r|fast3r|spann3r|mapanything|depth anything|learned|neural network|deep',re.I)
for p in d['all'].values():
    t=p['title']+' '+p['abstract']
    m=set(x.lower() for x in kw.findall(t))
    if m: print(p['id'], p['published'][:10], sorted(m), p['title'][:90])
