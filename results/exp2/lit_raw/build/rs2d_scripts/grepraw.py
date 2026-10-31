import urllib.request, re, sys
sys.stdout.reconfigure(encoding='utf-8')
# args: pattern then repo/branch/path ...
pat = re.compile(sys.argv[1], re.I)
for p in sys.argv[2:]:
    u = 'https://raw.githubusercontent.com/' + p
    try:
        t = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=40).read().decode('utf-8', 'replace')
    except Exception as e:
        print('==', p, 'ERR', e); continue
    print('==', p, len(t))
    for i, l in enumerate(t.splitlines()):
        if pat.search(l):
            print('  %4d: %s' % (i + 1, l.strip()[:220]))
