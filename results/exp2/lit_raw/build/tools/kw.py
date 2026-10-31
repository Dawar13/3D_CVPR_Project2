import sys, re
sys.stdout.reconfigure(encoding="utf-8")
fn=sys.argv[1]; pat=sys.argv[2]; w=int(sys.argv[3]) if len(sys.argv)>3 else 200; mx=int(sys.argv[4]) if len(sys.argv)>4 else 8
t=open(fn,encoding='utf-8').read()[3000:]
n=0
for m in re.finditer(pat,t,flags=re.I):
    print('...',t[max(0,m.start()-w):m.end()+w],'...\n'); n+=1
    if n>=mx: break
print('TOTAL matches:',len(re.findall(pat,t,flags=re.I)))
