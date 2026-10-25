# -*- coding: utf-8 -*-
"""Set a CJK font for the ablation figures so Chinese labels render."""
import io

P = r'E:\FINO\scripts\analyze_ablations.py'
s = io.open(P, encoding='utf-8').read()
if "Microsoft YaHei" in s:
    print('already patched')
else:
    old = "matplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    new = ("matplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
           "plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']\n"
           "plt.rcParams['axes.unicode_minus'] = False\n")
    assert old in s
    s = s.replace(old, new, 1)
    io.open(P, 'w', encoding='utf-8').write(s)
    print('font patched')
