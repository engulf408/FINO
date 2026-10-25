# -*- coding: utf-8 -*-
"""Second layout fix for slide 20: drop the FQL table row and move the footnote."""
import io

P = r'E:\FINO\ppt\build_fino_ppt.py'
s = io.open(P, encoding='utf-8').read()

old = """          [('antmaze-large', 'FINO', '1.00', '1.00', '0.98', '150.3 min / 31×50 eps'),
           ('humanoidmaze-medium', 'FINO', '1.00', '1.00', '0.96', '127.4 min / 16×20 eps*'),
           ('cube-double-play', 'FINO', '0.72', '0.92', '0.84', '144.2 min / 31×50 eps'),
           ('antmaze-large', 'FQL', '0.92', '0.98', '0.88', '136.6 min / 31×50 eps')],
          font_size=8.2, header_size=8.0, row_h=0.3)
text(s, 0.95, 5.98, 11.4, 0.45,
     '* humanoidmaze-medium 为节省时间采用每 100k 步 × 20 episodes；其余按论文协议（50k × 50）。全部为 seed=0，单 seed 结果仅看趋势。',
     size=7.8, color=GRAY)"""
new = """          [('antmaze-large', 'FINO', '1.00', '1.00', '0.98', '150.3 min / 31×50 eps'),
           ('humanoidmaze-medium', 'FINO', '1.00', '1.00', '0.96', '127.4 min / 16×20 eps*'),
           ('cube-double-play', 'FINO', '0.72', '0.92', '0.84', '144.2 min / 31×50 eps')],
          font_size=8.2, header_size=8.0, row_h=0.3)
text(s, 0.95, 6.18, 11.4, 0.45,
     '* humanoidmaze-medium 采用每 100k 步 × 20 episodes；其余按论文协议（50k × 50）。FQL 参考：antmaze-large 92%（峰值 98%）。全部 seed=0。',
     size=7.6, color=GRAY)"""
assert old in s
s = s.replace(old, new, 1)
# card a bit taller so the footnote is inside
s = s.replace("card(s, 0.7, 4.05, 11.93, 2.6)", "card(s, 0.7, 4.05, 11.93, 2.75)", 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('slide 20 second fix applied')
