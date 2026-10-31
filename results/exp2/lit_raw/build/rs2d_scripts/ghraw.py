import sys, re, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
H = {'User-Agent': 'Mozilla/5.0'}
def get(u):
    try:
        return urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40).read().decode('utf-8', 'replace')
    except Exception as e:
        return 'ERR ' + str(e)
full = '--full' in sys.argv
for r in [a for a in sys.argv[1:] if not a.startswith('--')]:
    print('=' * 10, r)
    rd = None; br = None
    for b in ['main', 'master']:
        for n in ['README.md', 'readme.md', 'Readme.md', 'README.MD']:
            t = get('https://raw.githubusercontent.com/%s/%s/%s' % (r, b, n))
            if not t.startswith('ERR'):
                rd, br = t, b; break
        if rd: break
    if not rd:
        print('no README found on main/master'); continue
    lic = 'none-found'
    for n in ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'license', 'License']:
        t = get('https://raw.githubusercontent.com/%s/%s/%s' % (r, br, n))
        if not t.startswith('ERR'):
            lic = n + ': ' + ' '.join(t.split()[:12]); break
    print('branch:', br, '| README chars:', len(rd), '| LICENSE:', lic)
    if full:
        print(rd[:6000]); continue
    for line in rd.splitlines():
        l = line.strip()
        if re.search(r'(drive\.google|huggingface|releases|pan\.baidu|sharepoint|onedrive|dropbox|zenodo|weight|checkpoint|ckpt|pretrain|pre-train|torch|pytorch|cuda|python|licen|dataset|coming soon|todo|release)', l, re.I):
            print('  |', l[:300])
