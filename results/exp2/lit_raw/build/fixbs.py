p = 'fill.py'
s = open(p, encoding='utf-8').read()
bs = chr(8)
s = s.replace(bs + 'MIT' + bs, chr(92) + 'bMIT' + chr(92) + 'b')
assert chr(8) not in s
open(p, 'w', encoding='utf-8').write(s)
