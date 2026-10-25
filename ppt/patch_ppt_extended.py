# -*- coding: utf-8 -*-
"""Add an extended-results slide to the PPT and update coverage/conclusion numbers."""
import io

P = r'E:\FINO\ppt\build_fino_ppt.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:90]
    s = s.replace(old, new, 1)


# ---------- coverage slide: 3 tasks completed ----------
rep("[('OGBench FINO（全预算）', '1 + 2 进行中', '1/35；含进行中 ≈ 8.6%'),",
    "[('OGBench FINO（全预算）', '3', '3/35 ≈ 8.6%'),")
rep("{'t': 'cube-double-play / humanoidmaze-medium 已跑 ~75%（FINO），预计 19:10 前后完成', 'bc': GREEN},",
    "{'t': '扩展任务已完成：cube-double-play 72%（峰值 92%）、humanoidmaze-medium 100%（FINO，全预算）', 'bc': GREEN},")
rep("{'t': '差距：45 任务中完整覆盖 1 个（≈2.2%）；补 3–5 任务 + 多 seed + 消融约需 50–100 GPU 小时，全量约 400–900 GPU 小时', 'bc': AMBER_D},",
    "{'t': '差距：45 任务中 FINO 覆盖 3 个（≈6.7%，含进行中 8.6%）；补 3–5 任务 + 多 seed + 消融约需 50–100 GPU 小时，全量约 400–900 GPU 小时', 'bc': AMBER_D},")

# ---------- conclusion slide: extended results ----------
rep("('✅ 数值级复现', 'antmaze-large 全预算：FINO 100% vs FQL 92%；单任务 2.3–2.5 h（GPU），方向与论文一致。', GREEN, GREEN_L),",
    "('✅ 数值级复现', '3 个 OGBench 任务全预算：antmaze-large 100%、humanoidmaze-medium 100%、cube-double-play 72%（FQL 对比 92%）。', GREEN, GREEN_L),")
rep("{'t': 'gpu_exp/：全预算 FINO/FQL 的 CSV、flags、1M 步 checkpoint'},",
    "{'t': 'gpu_exp/：4 条全预算运行（FINO×3、FQL×1）的 CSV/flags/checkpoint'},")

# ---------- new appendix slide before thanks ----------
marker = '# ================================================================ 19 thanks'
assert marker in s
new_slide = '''# ================================================================ 20 extended results
s = prs.slides.add_slide(BLANK)
add_header(s, '05 · 扩展结果', '扩展任务：FINO 在 3 个 OGBench 任务上的全预算表现', 20)
add_image(s, os.path.join(RES, 'fig_gpu_extended_curves.png'), 0.7, 1.5, 7.3, 2.65)
card(s, 8.2, 1.5, 4.43, 2.65, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 8.45, 1.63, 4.0, 0.3, '关键数字', size=10.5, bold=True, color=GREEN)
bullet_list(s, 8.45, 1.98, 4.0, 2.0, [
    {'t': 'antmaze-large：100%（FQL 92%）', 'bc': GREEN},
    {'t': 'humanoidmaze-medium：100%（峰值 100%）', 'bc': GREEN},
    {'t': 'cube-double-play：72%（峰值 92%，最后 5 次均值 84%）', 'bc': GREEN},
    {'t': '3 个任务全部在论文预算下完成（1M 离线 + 500k 在线）', 'bc': GREEN},
], size=9.2, gap=3)
card(s, 0.7, 4.32, 11.93, 2.35)
text(s, 0.95, 4.45, 11.4, 0.3, '全预算结果汇总（FINO vs FQL；每任务 1.5M 步）', size=10.5, bold=True, color=NAVY)
add_table(s, 0.95, 4.8, 11.35, [0.26, 0.10, 0.14, 0.12, 0.16, 0.22],
          ['任务', 'agent', '最终 success', '峰值', '最后 5 次均值', '耗时 / 评估协议'],
          [('antmaze-large', 'FINO', '1.00', '1.00', '0.98', '150.3 min / 31×50 eps'),
           ('humanoidmaze-medium', 'FINO', '1.00', '1.00', '0.96', '127.4 min / 16×20 eps*'),
           ('cube-double-play', 'FINO', '0.72', '0.92', '0.84', '144.2 min / 31×50 eps'),
           ('antmaze-large', 'FQL', '0.92', '0.98', '0.88', '136.6 min / 31×50 eps')],
          font_size=8.8, header_size=8.6, row_h=0.34)
text(s, 0.95, 6.32, 11.4, 0.3,
     '* humanoidmaze-medium 为节省时间采用每 100k 步 × 20 episodes；其余按论文协议（50k × 50）。'
     '全部为 seed=0，单 seed 结果仅看趋势。', size=8.4, color=GRAY)
add_footer(s, 20)
note(s, '扩展任务结果：FINO 在 humanoidmaze-medium 与 antmaze-large 都达到 100%，cube-double-play 72%（峰值 92%）；'
        '与 antmaze 的 FQL 对比一起，构成 3 个 OGBench 任务的全预算证据。')

'''
s = s.replace(marker, new_slide + marker, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('extended-results slide added, lines:', len(s.splitlines()))
