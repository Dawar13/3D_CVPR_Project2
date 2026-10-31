import sys, re, os, urllib.request, html as H, time
sys.stdout.reconfigure(encoding="utf-8")
out=sys.argv[1]; ids=sys.argv[2:]
for i in ids:
    fn=os.path.join(out,i+'.txt')
    if os.path.exists(fn) and os.path.getsize(fn)>5000: print(i,'cached',os.path.getsize(fn)); continue
    try:
        req=urllib.request.Request('https://arxiv.org/html/'+i, headers={'User-Agent':'Mozilla/5.0 (research lit review)'})
        d=urllib.request.urlopen(req,timeout=90).read().decode('utf-8','ignore')
    except Exception as e:
        print(i,'ERR',e); continue
    d=re.sub(r'<script.*?</script>|<style.*?</style>','',d,flags=re.S)
    t=H.unescape(re.sub(r'<[^>]+>',' ',d)); t=re.sub(r'\s+',' ',t)
    open(fn,'w',encoding='utf-8').write(t); print(i,'ok',len(t)); time.sleep(2)
