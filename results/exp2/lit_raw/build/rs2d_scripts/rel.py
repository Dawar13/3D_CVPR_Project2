import urllib.request, re, sys
sys.stdout.reconfigure(encoding='utf-8')
for u in sys.argv[1:]:
    try:
        t = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=40).read().decode('utf-8', 'replace')
    except Exception as e:
        print(u, 'ERR', e); continue
    print('==', u, len(t))
    for n in sorted(set(re.findall(r'/releases/download/[^"\s<>]+', t))):
        print('  asset', n)
    for n in sorted(set(re.findall(r'"path":"([^"]+)"', t)))[:80]:
        print('  path', n)
