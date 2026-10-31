import sys, json, re, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
H = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/vnd.github+json'}
def get(u):
    try:
        return urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40).read().decode('utf-8', 'replace')
    except Exception as e:
        return 'ERR ' + str(e)
for r in sys.argv[1:]:
    print('=' * 10, r)
    m = get('https://api.github.com/repos/' + r)
    if m.startswith('ERR'):
        print(m); continue
    d = json.loads(m)
    lic = (d.get('license') or {}).get('spdx_id')
    b = d.get('default_branch')
    print('license:', lic, '| branch:', b, '| pushed:', d.get('pushed_at'), '| stars:', d.get('stargazers_count'))
    rd = ''
    for n in ['README.md', 'readme.md', 'Readme.md']:
        rd = get('https://raw.githubusercontent.com/%s/%s/%s' % (r, b, n))
        if not rd.startswith('ERR'):
            break
    if rd.startswith('ERR'):
        print('no README'); continue
    print('README chars:', len(rd))
    for line in rd.splitlines():
        l = line.strip()
        if re.search(r'(drive\.google|huggingface|releases|pan\.baidu|sharepoint|onedrive|dropbox|zenodo|weight|checkpoint|ckpt|pretrain|pre-train|torch|pytorch|cuda|python|licen|dataset|coming soon|todo|release)', l, re.I):
            print('  |', l[:300])
