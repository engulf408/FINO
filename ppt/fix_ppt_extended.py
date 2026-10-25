# -*- coding: utf-8 -*-
"""Fix the layout of the extended-results slide (20)."""
import io

P = r'E:\FINO\ppt\build_fino_ppt.py'
s = io.open(P, encoding='utf-8').read()

start = s.index('# ================================================================ 20 extended results')
end = s.index('\n# ================================================================', start + 10)
new_block = '''# ================================================================ 20 extended results
s = prs.slides.add_slide(BLANK)
add_header(s, '05 · 扩展结果', '扩展任务：FINO 在 3 个 OGBench 任务上的全预算表现', 20)
add_image(s, os.path.join(RES, 'fig_gpu_extended_curves.png'), 0.7, 1.45, 7.3, 2.45)
card(s, 8.2, 1.45, 4.43, 2.45, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 8.45, 1.58, 4.0, 0.3, '关键数字', size=10.5, bold=True, color=GREEN)
bullet_list(s, 8.45, 1.92, 4.0, 1.85, [
    {'t': 'antmaze-large：100%（FQL 92%）', 'bc': GREEN},
    {'t': 'humanoidmaze-medium：100%（峰值 100%）', 'bc': GREEN},
    {'t': 'cube-double-play：72%（峰值 92%，最后 5 次均值 84%）', 'bc': GREEN},
    {'t': '3 个任务全部在论文预算下完成（1M 离线 + 500k 在线）', 'bc': GREEN},
], size=8.8, gap=2)
card(s, 0.7, 4.05, 11.93, 2.6)
text(s, 0.95, 4.16, 11.4, 0.28, '全预算结果汇总（FINO vs FQL；每任务 1.5M 步）', size=10.5, bold=True, color=NAVY)
add_table(s, 0.95, 4.5, 11.35, [0.26, 0.10, 0.14, 0.12, 0.16, 0.22],
          ['任务', 'agent', '最终 success', '峰值', '最后 5 次均值', '耗时 / 评估协议'],
          [('antmaze-large', 'FINO', '1.00', '1.00', '0.98', '150.3 min / 31×50 eps'),
           ('humanoidmaze-medium', 'FINO', '1.00', '1.00', '0.96', '127.4 min / 16×20 eps*'),
           ('cube-double-play', 'FINO', '0.72', '0.92', '0.84', '144.2 min / 31×50 eps'),
           ('antmaze-large', 'FQL', '0.92', '0.98', '0.88', '136.6 min / 31×50 eps')],
          font_size=8.2, header_size=8.0, row_h=0.3)
text(s, 0.95, 5.98, 11.4, 0.45,
     '* humanoidmaze-medium 为节省时间采用每 100k 步 × 20 episodes；其余按论文协议（50k × 50）。全部为 seed=0，单 seed 结果仅看趋势。',
     size=7.8, color=GRAY)
add_footer(s, 20)
note(s, '扩展任务结果：FINO 在 humanoidmaze-medium 与 antmaze-large 都达到 100%，cube-double-play 72%（峰值 92%）；'
        '与 antmaze 的 FQL 对比一起，构成 3 个 OGBench 任务的全预算证据。')

'''
s = s[:start] + new_block + s[end:]
io.open(P, 'w', encoding='utf-8').write(s)
print('slide 20 layout fixed')
