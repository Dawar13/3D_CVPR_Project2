import re, zlib, sys
sys.stdout.reconfigure(encoding='utf-8')
data = open(sys.argv[1], 'rb').read()
out = []
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
    s = m.group(1)
    try:
        d = zlib.decompress(s)
    except Exception:
        continue
    if b'Tj' in d or b'TJ' in d:
        parts = re.findall(rb'[(]([^)]*)[)]', d)
        txt = b''.join(parts).decode('latin-1')
        out.append(txt)
t = '\n'.join(out)
print(t[:int(sys.argv[2]) if len(sys.argv) > 2 else 4000])
