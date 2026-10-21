# -*- coding: utf-8 -*-
"""Fix page numbers on the coverage / conclusion slides."""
import io

p = r'E:\FINO\ppt\build_fino_ppt.py'
s = io.open(p, encoding='utf-8').read()
a = "add_header(s, '05 · 覆盖度', '论文 45 个任务：本次复现完成了多少？', 19)"
b = "add_header(s, '05 · 覆盖度', '论文 45 个任务：本次复现完成了多少？', 18)"
c = "add_header(s, '05 · 结论', '结论：代码级、机制级、数值级复现全部完成', 18)"
d = "add_header(s, '05 · 结论', '结论：代码级、机制级、数值级复现全部完成', 19)"
assert a in s and c in s
s = s.replace(a, b, 1).replace(c, d, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print('fixed header page numbers')
