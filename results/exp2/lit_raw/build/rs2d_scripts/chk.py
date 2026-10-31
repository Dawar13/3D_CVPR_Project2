import sys, re, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
# args: label|url ...  (reads at most 200 KB, writes nothing to disk)
for a in sys.argv[1:]:
    lab, u = a.split('|', 1)
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=40)
        body = r.read(200000)
        ct = r.headers.get('Content-Type', '')
        cl = r.headers.get('Content-Length', '')
        t = body.decode('utf-8', 'replace')
        m = re.search(r'<title>([^<]*)</title>', t)
        title = m.group(1).strip()[:100] if m else ''
        login = 'LOGIN-PAGE' if re.search(r'login\.microsoftonline|ServiceLogin|Sign in to your account', r.geturl() + title) else ''
        print('%s | %s | final=%s | type=%s | len=%s | title=%s %s' % (lab, r.status, r.geturl()[:120], ct[:40], cl, title, login))
    except Exception as e:
        print('%s | ERR %s' % (lab, e))
