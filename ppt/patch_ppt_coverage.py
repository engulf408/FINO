# -*- coding: utf-8 -*-
"""Insert an experiment-coverage slide (paper 45 tasks vs this reproduction) before the thanks slide."""
import io

P = r'E:\FINO\ppt\build_fino_ppt.py'
s = io.open(P, encoding='utf-8').read()

marker = '# ================================================================ 19 thanks'
assert marker in s
new_slide = '''# ================================================================ 19 coverage
s = prs.slides.add_slide(BLANK)
add_header(s, '05 · 覆盖度', '论文 45 个任务：本次复现完成了多少？', 19)
card(s, 0.7, 1.52, 5.5, 3.3)
text(s, 0.95, 1.66, 5.0, 0.3, '论文实验构成（README 45 条命令）', size=10.5, bold=True, color=NAVY)
add_table(s, 0.95, 2.02, 5.0, [0.46, 0.20, 0.34],
          ['类别', '数量', '说明'],
          [('OGBench 主任务', '35', '7 环境 × 5 task'),
           ('D4RL 任务', '10', 'antmaze/adroit/locomotion'),
           ('baseline 对比', '多个', 'FQL/IQL/IFQL/SAC/ReBRAC'),
           ('消融实验', '约 5 组', '噪声/Q-guided/β/flow steps/α')],
          font_size=8.8, header_size=8.8, row_h=0.42)
card(s, 6.45, 1.52, 6.18, 3.3)
text(s, 6.7, 1.66, 5.7, 0.3, '本复现实际覆盖（截至 10-21 18:20）', size=10.5, bold=True, color=NAVY)
add_table(s, 6.7, 2.02, 5.7, [0.44, 0.18, 0.38],
          ['类别', '完成', '覆盖率 / 备注'],
          [('OGBench FINO（全预算）', '1 + 2 进行中', '1/35；含进行中 ≈ 8.6%'),
           ('OGBench FQL（全预算）', '1', '1/35 ≈ 2.9%（antmaze-large）'),
           ('D4RL', '0', '0%（mujoco-py 未覆盖）'),
           ('论文消融', '机制级', '2D 玩具验证，非论文环境'),
           ('多 seed / 像素 / 视频', '0', '单 seed；无像素与视频')],
          font_size=8.6, header_size=8.6, row_h=0.42)
card(s, 0.7, 5.0, 11.93, 1.55, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 0.95, 5.13, 11.4, 0.3, '已完成的核心结果 & 差距', size=10.5, bold=True, color=GREEN)
bullet_list(s, 0.95, 5.48, 11.35, 1.0, [
    {'t': 'antmaze-large 全预算 FINO 100% vs FQL 92%（31 评估点 × 50 episodes）——论文主表同任务、同协议', 'bc': GREEN},
    {'t': 'cube-double-play / humanoidmaze-medium 已跑 ~75%（FINO），预计 19:10 前后完成', 'bc': GREEN},
    {'t': '差距：45 任务中完整覆盖 1 个（≈2.2%）；补 3–5 任务 + 多 seed + 消融约需 50–100 GPU 小时，全量约 400–900 GPU 小时', 'bc': AMBER_D},
], size=8.8, gap=2)
add_footer(s, 19)
note(s, '覆盖度盘点：论文 45 个任务中，本复现完整完成 1 个任务的主对比（FINO+FQL），另有 2 个 OGBench 任务接近完成，'
        'D4RL 与消融未覆盖。这是诚实的完成度口径。')

'''
s = s.replace(marker, new_slide + marker, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('coverage slide inserted, lines:', len(s.splitlines()))
