# -*- coding: utf-8 -*-
"""RL-course deck: FINO reproduction — results and experimental analysis (Chinese, 11 slides).

Output: E:\\FINO\\course\\RL课程_FINO复现_结果与分析.pptx
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

OUT = r'E:\FINO\course\RL课程_FINO复现_结果与分析.pptx'
RES = r'E:\FINO\results'

NAVY = RGBColor(0x0F, 0x1B, 0x2D)
INK = RGBColor(0x1B, 0x26, 0x3B)
GRAY = RGBColor(0x55, 0x61, 0x76)
LGRAY = RGBColor(0x8B, 0x96, 0xA8)
LINE = RGBColor(0xE2, 0xE7, 0xEE)
CARD = RGBColor(0xF5, 0xF7, 0xFB)
CARD2 = RGBColor(0xEC, 0xF1, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xF2, 0xA3, 0x3A)
AMBER_D = RGBColor(0xC9, 0x7F, 0x1C)
AMBER_L = RGBColor(0xFD, 0xF2, 0xE2)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
GREEN_L = RGBColor(0xE9, 0xF5, 0xEE)
BLUE = RGBColor(0x3B, 0x6F, 0xA0)
BLUE_L = RGBColor(0xE8, 0xEF, 0xF7)
RED = RGBColor(0xC0, 0x4B, 0x4B)
RED_L = RGBColor(0xFB, 0xEE, 0xEE)
FONT = 'Microsoft YaHei'
SW, SH = 13.333, 7.5
LM, CW = 0.75, 11.83


def set_font(run, size=13, bold=False, color=INK, spc=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = FONT
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ('a:ea', 'a:cs'):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set('typeface', FONT)
    if spc is not None:
        rPr.set('spc', str(int(spc * 100)))


def text(slide, x, y, w, h, s, size=13, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, line=1.18, name=FONT, spc=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line
    r = p.add_run()
    r.text = s
    set_font(r, size=size, bold=bold, color=color, spc=spc)
    return box


def rich(slide, x, y, w, h, parts, size=13, color=INK, line=1.18, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line
    for t, b, c in parts:
        r = p.add_run()
        r.text = t
        set_font(r, size=size, bold=b, color=c or color)
    return box


def rect(slide, x, y, w, h, fill=CARD, line_color=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            shp.adjustments[0] = radius
        except Exception:  # noqa: BLE001
            pass
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp


def card(slide, x, y, w, h, fill=CARD, line_color=LINE, radius=0.05):
    return rect(slide, x, y, w, h, fill=fill, line_color=line_color,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=radius)


def header(slide, tag, title, num):
    rect(slide, 0, 0, SW, 0.12, fill=NAVY)
    text(slide, LM, 0.38, 9.6, 0.28, tag, size=10.5, bold=True, color=AMBER_D, spc=1.4)
    text(slide, LM, 0.66, 11.6, 0.72, title, size=27, bold=True, color=NAVY)
    rect(slide, LM, 1.42, 0.55, 0.045, fill=AMBER)
    rect(slide, LM + 0.65, 1.44, CW - 0.65, 0.012, fill=LINE)
    text(slide, SW - 1.0, 0.38, 0.5, 0.28, f'{num:02d}', size=10.5, bold=True, color=LGRAY, align=PP_ALIGN.RIGHT)


def footer(slide, num, left='强化学习课程大作业 · FINO 复现'):
    rect(slide, LM, 7.0, CW, 0.01, fill=RGBColor(0xE7, 0xEB, 0xF2))
    text(slide, LM, 7.07, 9, 0.25, left, size=8.5, color=LGRAY)
    text(slide, SW - 1.15, 7.07, 0.5, 0.25, str(num), size=9.5, bold=True, color=LGRAY, align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=12, gap=5, color=INK, bc=AMBER_D):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = 1.22
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = '• '
        set_font(r, size=size, color=bc)
        r2 = p.add_run()
        r2.text = it
        set_font(r2, size=size, color=color)


def add_image(slide, path, x, y, max_w, max_h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    left, top = x + (max_w - w) / 2, y + (max_h - h) / 2
    slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(w), Inches(h))
    return left, top, w, h


def add_table(slide, x, y, w, ratios, header_row, rows, font_size=10, header_size=10,
              row_h=0.34, header_h=0.36, aligns=None):
    widths = [w * r for r in ratios]
    gf = slide.shapes.add_table(len(rows) + 1, len(header_row), Inches(x), Inches(y),
                                Inches(w), Inches(header_h + row_h * len(rows)))
    table = gf.table
    tblPr = table._tbl.tblPr
    for el in list(tblPr):
        if el.tag == qn('a:tableStyleId'):
            tblPr.remove(el)
    style = tblPr.makeelement(qn('a:tableStyleId'), {})
    style.text = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'
    tblPr.append(style)
    for i, wid in enumerate(widths):
        table.columns[i].width = Inches(wid)
    table.rows[0].height = Inches(header_h)
    for i in range(len(rows)):
        table.rows[i + 1].height = Inches(row_h)
    if aligns is None:
        aligns = [PP_ALIGN.LEFT] * len(header_row)

    def cell_style(cell, s, size, bold, color, align, fill):
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.margin_left = Inches(0.08)
        cell.margin_right = Inches(0.06)
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        p.line_spacing = 1.0
        r = p.add_run()
        r.text = str(s)
        set_font(r, size=size, bold=bold, color=color)

    for j, h in enumerate(header_row):
        cell_style(table.cell(0, j), h, header_size, True, WHITE,
                   PP_ALIGN.CENTER if j else PP_ALIGN.LEFT, NAVY)
    for i, row in enumerate(rows):
        fill = WHITE if i % 2 == 0 else CARD
        for j, v in enumerate(row):
            cell_style(table.cell(i + 1, j), v, font_size, j == 0, INK if j else NAVY, aligns[j], fill)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]

# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, fill=NAVY)
rect(s, 0.95, 1.05, 0.14, 0.14, fill=AMBER)
text(s, 1.2, 0.98, 10.5, 0.3, '强化学习课程大作业 · 论文复现', size=11, bold=True,
     color=RGBColor(0x9F, 0xB0, 0xC8), spc=1.6)
text(s, 0.95, 1.5, 11.8, 1.5, 'FINO 复现：结果与实验分析', size=40, bold=True, color=WHITE)
text(s, 0.95, 2.75, 11.0, 0.9,
     'Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning（ICLR 2026）\n'
     '官方实现复现：3 个 OGBench 任务全预算 + FQL 同预算对比',
     size=15, color=RGBColor(0xC3, 0xCF, 0xE0), line=1.45)
rect(s, 0.95, 4.05, 1.2, 0.04, fill=AMBER)
text(s, 0.95, 4.35, 11.5, 0.9,
     '核心数字：antmaze-large FINO 100% / FQL 92%；humanoidmaze-medium 100%；cube-double-play 72%（峰值 92%）。',
     size=17, bold=True, color=WHITE, line=1.35)
text(s, 0.95, 5.7, 11.0, 0.4, '汇报人：＿＿＿＿ · 课程：强化学习 · 2026-11', size=11,
     color=RGBColor(0x8F, 0xA6, 0xC2))
text(s, 0.95, 6.55, 11.0, 0.3, 'WSL2 + RTX 5070 · JAX 0.12(CUDA) · OGBench 1.1.0 · 仓库：github.com/engulf408/FINO',
     size=9.5, color=RGBColor(0x6F, 0x82, 0x9C))
notes(s, '开场：这是强化学习课程的论文复现。我复现了 ICLR 2026 的 FINO，按论文预算完整跑了 3 个 OGBench 任务，'
         '并做了同预算 FQL 对比；今天重点讲结果和实验分析。')

# ---------------------------------------------------------------- 2 goal
s = prs.slides.add_slide(BLANK)
header(s, '01 · 背景与目标', '复现目标：不是跑通，而是按论文协议做对比分析', 2)
card(s, 0.75, 1.7, 5.85, 2.5)
text(s, 1.0, 1.83, 5.4, 0.35, '离线到在线 RL 的难点', size=13.5, bold=True, color=NAVY)
bullets(s, 1.0, 2.2, 5.4, 1.9, [
    '离线阶段：策略不能访问数据外动作，Q 值容易高估',
    '在线阶段：只有少量交互预算，既要探索又要稳定',
    '目标：在保持性能的同时完成策略改进'
], size=11, gap=4)
card(s, 6.7, 1.7, 5.88, 2.5, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 6.95, 1.83, 5.4, 0.35, '为什么选 FINO 复现', size=13.5, bold=True, color=GREEN)
bullets(s, 6.95, 2.2, 5.4, 1.9, [
    'ICLR 2026 新方法：FQL + 噪声注入 + Q-guided/熵自适应 β',
    '官方仓库提供 OGBench 45 条完整命令，可对照',
    '我们具备 3 个任务的完整复现条件（数据/GPU/代码）'
], size=11, gap=4, bc=GREEN)
text(s, 0.75, 4.45, 11.8, 0.35, '本次复现的四项任务', size=13, bold=True, color=NAVY)
items = [('① 全预算主实验', 'antmaze-large FINO vs FQL'), ('② 扩展任务', 'humanoidmaze / cube-double'),
         ('③ 训练/评估分析', '曲线、损失、Q 值'), ('④ 机制验证', 'checkpoint 级 + 2D 实验')]
cw = (CW - 0.25 * 3) / 4
for i, (a, b) in enumerate(items):
    x = LM + i * (cw + 0.25)
    card(s, x, 4.9, cw, 1.25, fill=CARD2)
    text(s, x + 0.2, 5.03, cw - 0.35, 0.35, a, size=12, bold=True, color=AMBER_D)
    text(s, x + 0.2, 5.45, cw - 0.35, 0.55, b, size=10, color=INK, line=1.2)
text(s, 0.75, 6.35, 11.8, 0.4, '范围说明：论文主表 45 个任务中，本次完整覆盖 3 个 OGBench 任务（≈7%），其余为如实声明的局限。',
     size=10.5, color=GRAY)
footer(s, 2)
notes(s, '复现目标定得比"跑通"高：完整预算、同预算 baseline、训练/评估分析和机制验证。'
         '范围是3个OGBench任务，不是论文全量。')

# ---------------------------------------------------------------- 3 method
s = prs.slides.add_slide(BLANK)
header(s, '02 · 方法速览', 'FINO = FQL + 两处修改', 3)
mech = [
    ('① 噪声注入的 Flow Matching', '在插值点 x_t 上加时变高斯噪声（数据端方差 η²），扩大一步策略支持集', GREEN, GREEN_L),
    ('② 一步蒸馏', '学生策略逼近 10 步 flow 教师，推理从 10 步降到 1 步', BLUE, BLUE_L),
    ('③ Q-guided 采样 + 熵自适应 β', '在线采多个噪声候选按 softmax(βQ) 选；β 由策略熵反馈调节', AMBER_D, AMBER_L),
]
cw = (CW - 0.3 * 2) / 3
for i, (t, d, c, bg) in enumerate(mech):
    x = LM + i * (cw + 0.3)
    card(s, x, 1.75, cw, 1.95, fill=bg, line_color=c)
    text(s, x + 0.22, 1.9, cw - 0.4, 0.55, t, size=13.5, bold=True, color=c, line=1.2)
    text(s, x + 0.22, 2.5, cw - 0.42, 1.1, d, size=10.8, color=INK, line=1.3)
card(s, LM, 4.0, CW, 1.5)
text(s, 1.0, 4.13, 11.2, 0.35, '训练目标（一次 Adam 联合更新）', size=12.5, bold=True, color=NAVY)
rich(s, 1.0, 4.52, 11.2, 0.7,
     [('L = L_critic + L_actor,    L_actor = L_BC + α·L_distill + L_Q', True, INK)], size=15)
text(s, 1.0, 4.98, 11.2, 0.4,
     'L_BC：带噪 flow matching；L_distill：一步学生逼近 10 步教师；L_Q：−Q(s, 一步动作)。', size=10, color=GRAY)
text(s, 0.75, 5.8, 11.8, 0.85,
     '复现视角的关键结论：FINO 与 FQL 的唯一差别是噪声注入与 Q-guided/β；因此同预算 FINO vs FQL 可以干净地回答'
     '"这两处改动是否有效"。', size=11.5, color=INK, line=1.35)
footer(s, 3)
notes(s, '方法不是重点，一句话：FINO 在 FQL 上加噪声注入和 Q-guided+beta。'
         '这给复现一个很干净的设计：同预算 FINO vs FQL，唯一变量就是这两处。')

# ---------------------------------------------------------------- 4 implementation
s = prs.slides.add_slide(BLANK)
header(s, '03 · 实现', '复现环境与工程实现', 4)
add_table(s, LM, 1.75, CW, [0.22, 0.78],
          ['项目', '设置'],
          [('平台', 'Windows 11 + WSL2/Ubuntu 24.04；JAX 0.12(CUDA)；RTX 5070 12GB'),
           ('代码', '官方 FINO 代码；5 处平台/依赖补丁（4 处 Windows/CPU + 1 处 GPU distrax）；算法代码 0 改动'),
           ('数据', 'OGBench 1.1.0：antmaze-large / humanoidmaze-medium / cube-double-play（官方自动下载）'),
           ('训练开销', '单任务全预算 2.3–2.5 h（含 31 次评估）；4 条全预算运行约 9 GPU 小时'),
           ('可复现性', '代码/补丁/脚本/CSV/日志已整理；一条命令可在任意 Linux+GPU 机器重跑')],
          font_size=10.5, header_size=10.5, row_h=0.56)
card(s, LM, 5.3, 5.85, 1.35, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 1.0, 5.43, 5.4, 0.3, '为什么环境是个问题', size=11.5, bold=True, color=AMBER_D)
text(s, 1.0, 5.75, 5.4, 0.8,
     'JAX 的 CUDA 插件只有 Linux 轮子，Windows 原生只能 CPU（27 h/任务）→ 用 WSL2 打通 GPU，'
     '单任务降到 2.3–2.5 h。', size=9.8, color=INK, line=1.25)
card(s, 6.7, 5.3, 5.88, 1.35, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 6.95, 5.43, 5.4, 0.3, '为什么"算法零改动"重要', size=11.5, bold=True, color=GREEN)
text(s, 6.95, 5.75, 5.4, 0.8,
     'agents/fino.py 与官方仓库 SHA 完全一致；所有差异都在平台/依赖层 → 结果可归因于官方实现本身。',
     size=9.8, color=INK, line=1.25)
footer(s, 4)
notes(s, '实现：WSL2 + RTX5070 打通 JAX GPU；5 处平台/依赖补丁，算法代码和官方完全一致（SHA 相同）。'
         '单任务 2.3-2.5 小时。')

# ---------------------------------------------------------------- 5 setup
s = prs.slides.add_slide(BLANK)
header(s, '03 · 实验设置', '与论文一致的预算、评估协议与对比设计', 5)
card(s, LM, 1.7, 6.2, 2.85)
text(s, 1.0, 1.83, 5.7, 0.35, '实验配置', size=12.5, bold=True, color=NAVY)
add_table(s, 1.0, 2.2, 5.7, [0.42, 0.58],
          ['项目', '设置'],
          [('训练预算', '1M 离线 + 500k 在线'),
           ('评估', '每 50k 步 × 50 episodes，temperature=0'),
           ('任务', 'antmaze-large / humanoidmaze-medium / cube-double-play'),
           ('对比', 'FQL 同预算（仅 antmaze-large）'),
           ('随机性', 'seed=0（单 seed）')],
          font_size=9.2, header_size=9.2, row_h=0.34)
card(s, 7.15, 1.7, 5.43, 2.85, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 7.4, 1.83, 5.0, 0.35, '评估指标与预期', size=12.5, bold=True, color=AMBER_D)
bullets(s, 7.4, 2.2, 5.0, 2.2, [
    '主指标：最终成功率、最后 5 次均值、峰值',
    '学习曲线：success/return 随 update step 变化',
    '训练健康度：BC flow / 蒸馏 / critic 损失、Q 值',
    '机制分析：checkpoint 级（一步 vs 十步教师）'], size=9.8, gap=3, bc=AMBER_D)
card(s, LM, 4.75, CW, 1.5, fill=CARD2)
text(s, 1.0, 4.88, 11.2, 0.35, '与论文对齐的细节', size=12, bold=True, color=NAVY)
text(s, 1.0, 5.25, 11.2, 0.85,
     '超参按 README：antmaze-large α=10、q_agg=min；humanoidmaze-medium α=100、discount=0.995；cube-double-play α=300。'
     '评估 episodes 与间隔与论文一致（humanoidmaze 因时间成本改为 100k×20，已在结果中标注）。',
     size=10.2, color=INK, line=1.3)
footer(s, 5)
notes(s, '设置：1M+500k、50k×50 评估、三个任务、FQL 同预算、单 seed。这些和论文对齐，'
         '所以结果可以和论文同任务作量级对比。')

# ---------------------------------------------------------------- 6 main result
s = prs.slides.add_slide(BLANK)
header(s, '04 · 结果（一）', 'antmaze-large 全预算：FINO 100% vs FQL 92%', 6)
add_image(s, os.path.join(RES, 'fig_gpu_eval_antmaze-large-navigate-singletask-v0.png'), LM, 1.65, 7.35, 2.85)
card(s, 8.3, 1.65, 4.28, 2.85)
text(s, 8.55, 1.78, 3.8, 0.3, '关键数字', size=11.5, bold=True, color=NAVY)
rows = [('FINO 最终', '100%'), ('FQL 最终', '92%'), ('FINO 1M→1.5M', '0.82→1.00'), ('FQL 1M→1.5M', '0.78→0.92')]
yy = 2.2
for name, val in rows:
    text(s, 8.55, yy, 2.0, 0.3, name, size=9.5, color=GRAY)
    text(s, 10.6, yy - 0.02, 1.75, 0.35, val, size=12, bold=True, color=GREEN if 'FINO' in name else BLUE,
         align=PP_ALIGN.RIGHT)
    rect(s, 8.55, yy + 0.34, 3.75, 0.008, fill=LINE)
    yy += 0.5
card(s, LM, 4.75, 7.35, 1.55, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 1.0, 4.88, 6.9, 0.3, '实验分析', size=11.5, bold=True, color=GREEN)
text(s, 1.0, 5.23, 6.9, 0.95,
     'FINO 在离线期波动大于 FQL，但在线微调阶段反超：82%→100%（FQL 78%→92%）。'
     '说明噪声注入带来的支持集扩展在在线阶段产生了收益。', size=10.2, color=INK, line=1.3)
card(s, 8.3, 4.75, 4.28, 1.55)
text(s, 8.55, 4.88, 3.8, 0.3, '运行成本', size=11, bold=True, color=NAVY)
add_table(s, 8.55, 5.2, 3.8, [0.46, 0.27, 0.27],
          ['', 'FINO', 'FQL'],
          [('总耗时', '150.3 min', '136.6 min'), ('平均速度', '6.01 ms', '5.47 ms')],
          font_size=8.6, header_size=8.6, row_h=0.34)
footer(s, 6)
notes(s, '主结果：antmaze-large 全预算 FINO 100%、FQL 92%。关键分析：FINO 离线期波动大，'
         '但在线微调阶段从82%到100%反超，说明噪声注入的支持集扩展在在线阶段起作用。')

# ---------------------------------------------------------------- 7 extended
s = prs.slides.add_slide(BLANK)
header(s, '04 · 结果（二）', '扩展到 3 个 OGBench 任务：均学到有效策略', 7)
add_table(s, LM, 1.75, 6.2, [0.38, 0.14, 0.18, 0.14, 0.16],
          ['任务', 'agent', '最终 success', '峰值', 'last-5'],
          [('antmaze-large', 'FINO', '1.00', '1.00', '0.98'),
           ('antmaze-large', 'FQL', '0.92', '0.98', '0.88'),
           ('humanoidmaze-medium', 'FINO', '1.00', '1.00', '0.96'),
           ('cube-double-play', 'FINO', '0.72', '0.92', '0.84')],
          font_size=9.6, header_size=9.2, row_h=0.4,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
add_image(s, os.path.join(RES, 'fig_gpu_task_summary.png'), 7.1, 1.7, 5.45, 3.35)
card(s, LM, 4.35, CW, 2.0, fill=CARD2)
text(s, 1.0, 4.48, 11.2, 0.32, '结果分析', size=12, bold=True, color=NAVY)
bullets(s, 1.0, 4.85, 11.2, 1.45, [
    '两个任务（antmaze-large、humanoidmaze-medium）达到 100%，说明方法可迁移到不同形态的任务',
    'cube-double-play 最终 72%、峰值 92%、last-5 均值 84%，波动较大（单 seed），但显著优于随机',
    '任务难度不同：cube-double-play 是多物体操作任务、评估噪声更大；humanoidmaze 是高维人形控制'
], size=10, gap=3)
footer(s, 7)
notes(s, '三个任务：两个100%，cube-double 72%、峰值92%。分析：方法可迁移，但 cube-double 波动大；'
         '不同任务难度和评估噪声不同。')

# ---------------------------------------------------------------- 8 training analysis
s = prs.slides.add_slide(BLANK)
header(s, '05 · 实验分析（一）', '训练过程：损失正常收敛，Q 值趋向理论下限', 8)
add_image(s, os.path.join(RES, 'fig_gpu_training_FINO_antmaze-large-navigate-singletask-v0.png'),
          LM, 1.65, 6.1, 4.2)
card(s, 7.05, 1.65, 5.5, 2.6)
text(s, 7.3, 1.78, 5.0, 0.3, '关键指标（起点 → 1.5M 步）', size=11, bold=True, color=NAVY)
add_table(s, 7.3, 2.15, 5.0, [0.38, 0.30, 0.32],
          ['指标', 'FINO', '判断'],
          [('BC flow loss', '0.451→0.285', '↓ 正常'),
           ('一步蒸馏 loss', '0.0198→0.0151', '↓ 正常'),
           ('critic TD loss', '0.183→0.212', '稳定'),
           ('策略动作 Q 均值', '−24.2→−90.7', '趋近 −100')],
          font_size=9.2, header_size=9.0, row_h=0.4)
card(s, 7.05, 4.45, 5.5, 1.4, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 7.3, 4.57, 5.0, 0.3, '分析结论', size=11, bold=True, color=GREEN)
text(s, 7.3, 4.9, 5.0, 0.85,
     '损失全程下降、Q 收敛到 −90 附近（理论下限 −100）→ 训练实现正确；性能差异不是"训练不收敛"造成的。',
     size=9.8, color=INK, line=1.3)
text(s, LM, 6.2, 11.8, 0.5,
     '注：曲线为 FINO 在 antmaze-large 的 4 联图（BC flow / 蒸馏 / critic / Q）；虚线为 1M 步离线→在线切换点。',
     size=9.5, color=GRAY)
footer(s, 8)
notes(s, '训练分析：BC flow 和蒸馏损失下降、Q 从 -24 收敛到 -90（理论下限 -100），说明训练实现正确，'
         '性能不是训练不收敛导致的。')

# ---------------------------------------------------------------- 9 mechanism analysis
s = prs.slides.add_slide(BLANK)
header(s, '05 · 实验分析（二）', '机制与 checkpoint 分析：蒸馏保真、策略拟合充分', 9)
add_table(s, LM, 1.75, 7.0, [0.44, 0.14, 0.14, 0.28],
          ['指标（1M 步 checkpoint）', 'FINO', 'FQL', '含义'],
          [('一步策略 vs 10 步教师 ED', '0.0061', '0.0055', '蒸馏保真'),
           ('一步策略 vs 数据集动作 MSE', '0.0901', '0.0904', '拟合改善'),
           ('10 步教师 vs 数据集动作 MSE', '0.0595', '0.0602', '优于 30k 步时 0.077'),
           ('贪心策略平均 Q', '−92.1', '−92.0', '接近理论下限')],
          font_size=9.4, header_size=9.0, row_h=0.45)
card(s, 7.85, 1.75, 4.73, 2.75, fill=BLUE_L, line_color=RGBColor(0xC9, 0xD9, 0xEA))
text(s, 8.1, 1.88, 4.2, 0.3, '2D 玩具交叉验证', size=11.5, bold=True, color=BLUE)
bullets(s, 8.1, 2.25, 4.2, 2.1, [
    '一步 vs 十步教师 EMD 0.0004',
    '推理 1.44 ms vs 7.77 ms（5.38×）',
    'Q-guided d=6 时 Q +0.63；d=2 退化',
    'β 自适应方向与代码一致'], size=9.4, gap=3, bc=BLUE)
card(s, LM, 4.75, CW, 1.6, fill=CARD2)
text(s, 1.0, 4.88, 11.2, 0.32, '分析结论', size=11.5, bold=True, color=NAVY)
text(s, 1.0, 5.25, 11.2, 0.95,
     '① 一步蒸馏在高维任务上仍保真（ED≈0.006）；② 教师与策略对数据动作的 MSE 比小预算时下降约 25%，全预算下拟合充分；'
     '③ 平均 Q≈−92 与高成功率自洽，说明 critic 与策略互相匹配。', size=10.2, color=INK, line=1.35)
footer(s, 9)
notes(s, '机制分析：checkpoint 上一步蒸馏 ED 0.006，教师对数据的 MSE 从 30k步的0.077降到0.060，'
         'Q 接近下限；2D 玩具交叉验证了蒸馏速度和 Q-guided/β 的方向。')

# ---------------------------------------------------------------- 10 formula-logic ablation
s = prs.slides.add_slide(BLANK)
header(s, '05 · 原创消融', '公式逻辑验证：论文公式 vs 实现（antmaze-large，全预算）', 10)
card(s, LM, 1.52, 6.1, 2.05)
text(s, 1.0, 1.62, 5.6, 0.28, 'A. 噪声 schedule（从零 1M+500k）', size=10.8, bold=True, color=NAVY)
add_table(s, 1.0, 1.94, 5.6, [0.46, 0.18, 0.18, 0.18],
          ['变体', '最终', '峰值', 'last-5'],
          [('官方指数 η=0.1', '1.00', '1.00', '0.98'),
           ('无噪声 η=0', '0.96', '0.98', '0.88'),
           ('指数 η=0.3', '0.96', '0.96', '0.75'),
           ('论文二次型 Eq.(7)', '0.04', '0.62', '0.01')],
          font_size=8.2, header_size=8.2, row_h=0.26,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
card(s, 6.95, 1.52, 5.63, 2.05)
text(s, 7.2, 1.62, 5.1, 0.28, 'B. β 更新（1M ckpt 起在线 500k）', size=10.8, bold=True, color=NAVY)
add_table(s, 7.2, 1.94, 5.1, [0.44, 0.18, 0.18, 0.20],
          ['变体', '最终', '峰值', 'β 末值'],
          [('官方 β', '0.94', '1.00', '12.9'),
           ('论文 Eq.(9) 符号', '0.96', '1.00', '7.1'),
           ('固定 β=10', '0.94', '1.00', '10.0')],
          font_size=8.2, header_size=8.2, row_h=0.28,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
add_image(s, os.path.join(RES, 'fig_ablation_schedule_success.png'), LM, 3.68, 6.35, 1.9)
card(s, 7.35, 3.68, 5.23, 1.9, fill=RED_L, line_color=RGBColor(0xE8, 0xC4, 0xC4))
text(s, 7.55, 3.77, 4.8, 0.28, '分析', size=10.8, bold=True, color=RED)
bullets(s, 7.55, 4.07, 4.8, 1.5, [
    '论文 Eq.(7) 字面实现：在线期崩溃（1M 时 0.62 → 1.1M 后 0.02–0）',
    'BC flow loss 0.42 vs 官方 0.29 / 无噪声 0.26 → 中段噪声（约 340×）破坏拟合',
    '官方指数 schedule 最优；无噪声、η=0.3 均 0.96 → 噪声不是越强越好',
    'β 符号方向影响很小（0.94 vs 0.96）；固定 β 略低'], size=8.2, gap=2, bc=RED)
text(s, LM, 5.72, CW, 0.9,
     '结论：论文公式与实现不一致不仅是文档问题——按 Eq.(7) 实现会破坏训练（0.04 vs 1.00）；'
     '以实现为准的复现策略成立，同时说明论文的公式逻辑需要修正。单 seed，但效应量级差异显著。',
     size=10.2, color=INK, line=1.3)
footer(s, 10)
notes(s, '公式逻辑消融：论文二次型 schedule 按字面实现会崩溃（0.04），官方指数 schedule 最优（1.00）；'
         '无噪声和 eta=0.3 都是 0.96，说明噪声不是越强越好；beta 符号方向影响很小（0.94 vs 0.96）。')

# ---------------------------------------------------------------- 10 coverage & issues
s = prs.slides.add_slide(BLANK)
header(s, '06 · 覆盖面与问题', '覆盖 3/45 任务；复现中发现 4 个实现问题', 11)
card(s, LM, 1.7, 5.85, 2.6)
text(s, 1.0, 1.83, 5.4, 0.35, '覆盖面（如实说明）', size=12.5, bold=True, color=NAVY)
add_table(s, 1.0, 2.2, 5.35, [0.50, 0.20, 0.30],
          ['类别', '规模', '本复现'],
          [('OGBench 主任务', '35', '3 完成'),
           ('FQL 对比', '35', '1'),
           ('D4RL 任务', '10', '0'),
           ('消融 / 多 seed', '多组', '未做')],
          font_size=9.2, header_size=9.0, row_h=0.33)
text(s, 1.0, 4.02, 5.35, 0.3, '合计完成：3/45 ≈ 7%（单 seed）', size=10, color=AMBER_D, bold=True)
card(s, 6.7, 1.7, 5.88, 2.6, fill=RED_L, line_color=RGBColor(0xE8, 0xC4, 0xC4))
text(s, 6.95, 1.83, 5.4, 0.35, '复现中发现的实现问题（有助于解释结果）', size=11.5, bold=True, color=RED)
bullets(s, 6.95, 2.2, 5.4, 2.0, [
    '噪声注入 schedule：代码指数型 vs 论文 Eq.(7) 二次型',
    'β 更新符号：代码与论文 Eq.(9) 印刷方向相反',
    'to_eval 全局 argmax：batch>1 时跨状态污染',
    'CsvLogger/β 存档：在线指标不写 CSV、β 不随 checkpoint 保存'], size=9.2, gap=2, bc=RED)
card(s, LM, 4.55, CW, 1.8, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 1.0, 4.68, 11.2, 0.32, '对复现结论的影响', size=11.5, bold=True, color=AMBER_D)
text(s, 1.0, 5.05, 11.2, 1.1,
     '这些是"论文描述与实现不一致"或"实验记录缺陷"，不影响主结果的成立（我们以代码为准复现）；'
     '但说明：① 若要严格对齐论文公式，需要做 schedule/β 的消融；② 单 seed + 3/45 覆盖，只能下"方法有效、与论文同量级"的结论，不能下"完整复现论文"的结论。',
     size=10.2, color=INK, line=1.35)
footer(s, 11)
notes(s, '覆盖面和问题：完成3/45任务、单seed；复现中发现4个实现问题（schedule、beta符号、batch argmax、日志）。'
         '这些不影响主结果，但限制了结论的强度。')

# ---------------------------------------------------------------- 11 conclusion
s = prs.slides.add_slide(BLANK)
header(s, '07 · 结论', '结论与展望', 12)
cards = [
    ('✅ 复现结论', '按论文预算复现 3 个 OGBench 任务：antmaze-large FINO 100% vs FQL 92%、humanoidmaze 100%、cube-double 72%；'
                '在线微调阶段收益显著（82%→100%）。', GREEN, GREEN_L),
    ('🔍 分析结论', '训练损失与 Q 值收敛正常；蒸馏保真（ED≈0.006）；公式消融显示论文 Eq.(7) 字面实现会崩溃'
                 '（0.04），官方实现最优（1.00），β 符号影响很小（0.94 vs 0.96）。', BLUE, BLUE_L),
    ('⚠️ 局限', '3/45 任务、单 seed、无 D4RL；公式消融已补做（A/B 组），Q-guided K 与多 seed 未做；'
              '主结果仍只能下"方法有效、与论文同量级"的结论。', AMBER_D, AMBER_L),
]
for i, (t, d, c, bg) in enumerate(cards):
    y = 1.75 + i * 1.55
    card(s, LM, y, CW, 1.35, fill=bg, line_color=c)
    text(s, 1.0, y + 0.13, 3.6, 0.4, t, size=14.5, bold=True, color=c)
    text(s, 4.3, y + 0.13, 8.0, 1.1, d, size=10.5, color=INK, line=1.3)
text(s, LM, 6.55, 8.5, 0.35, '仓库与数据：github.com/engulf408/FINO', size=11, color=GRAY)
text(s, SW - 5.4, 6.55, 5.0, 0.35, '谢谢 / Q&A', size=12, bold=True, color=AMBER_D, align=PP_ALIGN.RIGHT)
footer(s, 12)
notes(s, '结论三块：复现结果、分析结论、局限。仓库在 GitHub，谢谢，欢迎提问。')

prs.core_properties.title = 'FINO 复现：结果与实验分析（强化学习课程大作业）'
prs.core_properties.author = '＿＿＿＿'
prs.save(OUT)
print('saved:', OUT, '| slides:', len(prs.slides._sldIdLst))
