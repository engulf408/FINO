# -*- coding: utf-8 -*-
"""Fix overlap on RL deck slide 10 (ablation tables overlapping the figure)."""
import io

P = r'E:\FINO\course\build_rl_deck.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:100]
    s = s.replace(old, new, 1)


rep("card(s, LM, 1.6, 6.1, 1.75)", "card(s, LM, 1.52, 6.1, 2.05)")
rep("text(s, 1.0, 1.7, 5.6, 0.28, 'A. 噪声 schedule（从零 1M+500k）', size=10.8, bold=True, color=NAVY)",
    "text(s, 1.0, 1.62, 5.6, 0.28, 'A. 噪声 schedule（从零 1M+500k）', size=10.8, bold=True, color=NAVY)")
rep("add_table(s, 1.0, 2.02, 5.6, [0.46, 0.18, 0.18, 0.18],",
    "add_table(s, 1.0, 1.94, 5.6, [0.46, 0.18, 0.18, 0.18],")
rep("          font_size=8.4, header_size=8.4, row_h=0.28,",
    "          font_size=8.2, header_size=8.2, row_h=0.26,")
rep("card(s, 6.95, 1.6, 5.63, 1.75)", "card(s, 6.95, 1.52, 5.63, 2.05)")
rep("text(s, 7.2, 1.7, 5.1, 0.28, 'B. β 更新（1M ckpt 起在线 500k）', size=10.8, bold=True, color=NAVY)",
    "text(s, 7.2, 1.62, 5.1, 0.28, 'B. β 更新（1M ckpt 起在线 500k）', size=10.8, bold=True, color=NAVY)")
rep("add_table(s, 7.2, 2.02, 5.1, [0.44, 0.18, 0.18, 0.20],",
    "add_table(s, 7.2, 1.94, 5.1, [0.44, 0.18, 0.18, 0.20],")
rep("          font_size=8.4, header_size=8.4, row_h=0.3,",
    "          font_size=8.2, header_size=8.2, row_h=0.28,")
rep("add_image(s, os.path.join(RES, 'fig_ablation_schedule_success.png'), LM, 3.42, 6.35, 2.12)",
    "add_image(s, os.path.join(RES, 'fig_ablation_schedule_success.png'), LM, 3.68, 6.35, 1.9)")
rep("card(s, 7.35, 3.42, 5.23, 2.12, fill=RED_L, line_color=RGBColor(0xE8, 0xC4, 0xC4))",
    "card(s, 7.35, 3.68, 5.23, 1.9, fill=RED_L, line_color=RGBColor(0xE8, 0xC4, 0xC4))")
rep("text(s, 7.55, 3.52, 4.8, 0.28, '分析', size=10.8, bold=True, color=RED)",
    "text(s, 7.55, 3.77, 4.8, 0.28, '分析', size=10.8, bold=True, color=RED)")
rep("bullets(s, 7.55, 3.84, 4.8, 1.6, [", "bullets(s, 7.55, 4.07, 4.8, 1.5, [")
rep("text(s, LM, 5.75, CW, 0.9,", "text(s, LM, 5.72, CW, 0.9,")

io.open(P, 'w', encoding='utf-8').write(s)
print('slide 10 geometry fixed')
