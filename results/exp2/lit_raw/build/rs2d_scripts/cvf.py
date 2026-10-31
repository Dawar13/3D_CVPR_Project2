import sys,re,html,urllib.request
sys.stdout.reconfigure(encoding="utf-8")
for u in sys.argv[1:]:
    try:
        req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        t=urllib.request.urlopen(req,timeout=40).read().decode('utf-8','replace')
    except Exception as e:
        print('=== FAIL',u,e); continue
    print('=== OK',u)
    for k in ['papertitle','authors']:
        m=re.search(r'<div id="'+k+r'">(.*?)</div>',t,re.S)
        if m: print(' ',k,':',' '.join(html.unescape(re.sub('<[^>]+>','',m.group(1))).split()))
    m=re.search(r'id="abstract"[^>]*>(.*?)</div>',t,re.S)
    if m: print('  abstract:',' '.join(html.unescape(re.sub('<[^>]+>','',m.group(1))).split())[:1800])
