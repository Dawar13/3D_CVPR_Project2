import re,sys,html
sys.stdout.reconfigure(encoding="utf-8")
t=open(sys.argv[1],encoding='utf-8',errors='replace').read()
t=re.sub(r'<script.*?</script>|<style.*?</style>','',t,flags=re.S|re.I)
t=html.unescape(re.sub(r'<[^>]+>',' ',t))
t=' '.join(t.split())
s=int(sys.argv[2]) if len(sys.argv)>2 else 0
n=int(sys.argv[3]) if len(sys.argv)>3 else 3000
print(t[s:s+n])
