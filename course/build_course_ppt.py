# -*- coding: utf-8 -*-
"""Build a 10-slide (10-minute) Chinese deck skeleton for the optimization-course assignment.

Topic: FINO as a generative policy optimization method — reproduction, audit and mechanism ablations.
Ablation results are placeholders (to be filled after the experiments).

Output: E:\\FINO\\course\\FINO_优化方法_10分钟汇报_骨架.pptx
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

OUT_DIR = r'E:\FINO\course'
OUT = os.path.join(OUT_DIR, '优化课程_FINO优化方法介绍.pptx')
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


def set_font(run, size=13, bold=False, color=INK, name=FONT, spc=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ('a:ea', 'a:cs'):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set('typeface', name)
    if spc is not None:
        rPr.set('spc', str(int(spc * 100)))


def text(slide, x, y, w, h, s, size=13, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, line=1.18, spc=None):
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


def rich(slide, x, y, w, h, parts, size=13, color=INK, line=1.18):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.line_spacing = line
    for t, b, c in parts:
        r = p.add_run()
        r.text = t
        set_font(r, size=size, bold=b, color=c or color)
    return box


def rect(slide, x, y, w, h, fill=CARD, line_color=None, shape=MSO_SHAPE.RECTANGLE, radius=None, dash=None):
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
        shp.line.width = Pt(1.0)
        if dash is not None:
            shp.line.dash_style = dash
    shp.shadow.inherit = False
    return shp


def card(slide, x, y, w, h, fill=CARD, line_color=LINE):
    return rect(slide, x, y, w, h, fill=fill, line_color=line_color,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)


def placeholder(slide, x, y, w, h, label='【待补：原创消融结果】'):
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    rect(slide, x, y, w, h, fill=RGBColor(0xFA, 0xFB, 0xFD), line_color=LGRAY,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04, dash=MSO_LINE_DASH_STYLE.DASH)
    text(slide, x + 0.15, y + 0.12, w - 0.3, h - 0.24, label, size=12, color=LGRAY,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def header(slide, tag, title, num):
    rect(slide, 0, 0, SW, 0.12, fill=NAVY)
    text(slide, LM, 0.38, 9, 0.28, tag, size=10.5, bold=True, color=AMBER_D, spc=1.4)
    text(slide, LM, 0.66, 11.6, 0.72, title, size=27, bold=True, color=NAVY)
    rect(slide, LM, 1.42, 0.55, 0.045, fill=AMBER)
    rect(slide, LM + 0.65, 1.44, CW - 0.65, 0.012, fill=LINE)
    text(slide, SW - 1.0, 0.38, 0.5, 0.28, f'{num:02d}', size=10.5, bold=True, color=LGRAY, align=PP_ALIGN.RIGHT)


def footer(slide, num, text_left='FINO 优化方法研究 · 优化方法课程报告'):
    rect(slide, LM, 7.0, CW, 0.01, fill=RGBColor(0xE7, 0xEB, 0xF2))
    text(slide, LM, 7.07, 9, 0.25, text_left, size=8.5, color=LGRAY)
    text(slide, SW - 1.15, 7.07, 0.5, 0.25, str(num), size=9.5, bold=True, color=LGRAY, align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=12.5, gap=5, color=INK, bullet_color=AMBER_D):
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
        set_font(r, size=size, color=bullet_color)
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

# ------------------------------------------------------------------ 1 cover
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, fill=NAVY)
rect(s, 0.95, 1.05, 0.14, 0.14, fill=AMBER)
text(s, 1.2, 0.98, 9.5, 0.3, '优化方法课程 · TOPIC 1：一种新的优化方法', size=11, bold=True,
     color=RGBColor(0x9F, 0xB0, 0xC8), spc=1.6)
text(s, 0.95, 1.5, 11.6, 1.5,
     'FINO：面向离线到在线强化学习的\n生成式策略优化方法', size=36, bold=True, color=WHITE, line=1.25)
rect(s, 0.95, 3.35, 1.2, 0.04, fill=AMBER)
text(s, 0.95, 3.65, 11.0, 0.5,
     '原理 · 实现 · 案例研究 · 原创审计与机制消融',
     size=18, color=RGBColor(0xC3, 0xCF, 0xE0))
text(s, 0.95, 4.5, 11.0, 1.0,
     '复现 + 审计：3 个 OGBench 任务按论文预算运行，antmaze-large 上 FINO 100% vs FQL 92%；\n'
     '发现论文噪声 schedule 与实现不一致、β 更新符号与论文相反，并设计真实任务机制消融。',
     size=13, color=WHITE, line=1.5)
text(s, 0.95, 6.35, 11.0, 0.3, '汇报人：＿＿＿＿   ·   课程：＿＿＿＿   ·   2026-11', size=10.5,
     color=RGBColor(0x8F, 0xA6, 0xC2))
notes(s, '开场：选题是"一种新的优化方法"。我把 FINO 当作生成式策略优化器来研究：先讲它解决什么优化问题、'
         '怎么实现，再用真实任务做案例，最后给出我自己的审计与原创消融设计。')

# ------------------------------------------------------------------ 2 motivation
s = prs.slides.add_slide(BLANK)
header(s, '01 · 动机', '离线到在线 RL 的核心是一个受限优化问题', 2)
fails = [
    ('① 分布外高估', '离线数据覆盖不到的动作被 Q 网络高估 → 在线一执行就崩'),
    ('② 过度保守', '只敢贴着数据动作 → 在线探索不足，微调收益小'),
    ('③ 策略梯度不稳', 'on-policy 梯度方差大，少量在线交互很难稳定优化'),
]
cw = (CW - 0.3 * 2) / 3
for i, (t, d) in enumerate(fails):
    x = LM + i * (cw + 0.3)
    card(s, x, 1.75, cw, 1.95, fill=RED_L if i < 2 else AMBER_L,
         line_color=RGBColor(0xE8, 0xC4, 0xC4) if i < 2 else RGBColor(0xF3, 0xD8, 0xB0))
    text(s, x + 0.22, 1.88, cw - 0.4, 0.35, t, size=14.5, bold=True, color=RED if i < 2 else AMBER_D)
    text(s, x + 0.22, 2.3, cw - 0.42, 1.25, d, size=11.5, color=INK, line=1.3)
card(s, LM, 3.95, CW, 1.55, fill=CARD2)
text(s, 1.0, 4.08, 5.0, 0.35, '优化视角的统一形式', size=13.5, bold=True, color=NAVY)
rich(s, 1.0, 4.45, 11.2, 0.9,
     [('max_π  E[Q(s, π(s))]   s.t.   D(π(·|s) ‖ μ(·|s)) ≤ ε', True, NAVY)], size=15)
rich(s, 1.0, 4.9, 11.2, 0.5,
     [('不同算法 = 对约束 D、权重 α 与优化器的不同选择；FINO 用"生成式策略 + 噪声正则 + 对偶探索"来解。',
       False, GRAY)], size=11.5)
text(s, LM, 5.85, CW, 0.35, '为什么选 FINO 作为"新优化方法"来介绍', size=13.5, bold=True, color=NAVY)
bullets(s, LM, 6.25, CW, 0.7, ['2026 年 ICLR 新方法（FQL 之上两处优化设计）；把策略优化变成"回归 + 蒸馏 + 对偶更新"的混合优化问题，'
                               '适合作为优化课程的方法剖析；我们已具备完整复现与 GPU 条件，可做真实任务上的原创消融。'], size=10.5)
footer(s, 2)
notes(s, '动机：离线到在线 RL 有三个典型失败模式，本质是一个带约束的优化问题。FINO 是 2026 年新方法，'
         '把生成式策略、噪声正则和对偶式探索结合起来，值得从优化角度研究。')

# ------------------------------------------------------------------ 3 formalization
s = prs.slides.add_slide(BLANK)
header(s, '02 · 问题形式化', '把生成式策略学习写成优化问题', 3)
card(s, LM, 1.7, 5.85, 2.4)
text(s, 1.0, 1.83, 5.4, 0.35, '① 带约束的策略优化', size=13.5, bold=True, color=NAVY)
rich(s, 1.0, 2.25, 5.4, 1.5,
     [('max_θ  E_{s∼dπ}[ Q^π(s, π_θ(s)) ]\n'
       's.t.  D( π_θ(·|s) ‖ μ(·|s) ) ≤ ε', True, INK)], size=13.5, line=1.5)
text(s, 1.0, 3.55, 5.4, 0.45, 'Lagrange 形式：max_π min_β E[Q] + β(ε − D) → β 是对偶变量', size=10.5, color=GRAY)
card(s, 6.7, 1.7, 5.85, 2.4)
text(s, 6.95, 1.83, 5.4, 0.35, '② Flow Matching 作为回归优化', size=13.5, bold=True, color=NAVY)
rich(s, 6.95, 2.25, 5.4, 1.5,
     [('L_CFM(θ) = E_{t,x0,x1}[ ‖v_θ(x_t,t) − (x1−x0)‖² ]\n'
       'x_t = (1−t)x0 + t·x1,  x0∼N(0,I),  x1∼π_data', True, INK)], size=13, line=1.5)
text(s, 6.95, 3.55, 5.4, 0.45, '沿概率路径的随机最小二乘；推理 = Euler 积分（K=10 步）', size=10.5, color=GRAY)
card(s, LM, 4.35, CW, 1.55, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 1.0, 4.48, 11.2, 0.35, '③ FINO 的单步训练目标（一次 Adam 联合更新）', size=13, bold=True, color=AMBER_D)
rich(s, 1.0, 4.9, 11.2, 0.8,
     [('L = L_critic + L_actor ,    L_actor = L_BC + α·L_distill + L_Q', True, INK)], size=16)
rich(s, 1.0, 5.45, 11.2, 0.4,
     [('L_critic：TD 误差；L_BC：带噪插值点的 CFM；L_distill：一步学生逼近 10 步教师；L_Q：−Q(s, 一步动作)。',
       False, GRAY)], size=10.8)
text(s, LM, 6.15, CW, 0.5,
     '优化结构：不是交替优化，而是"监督回归 + 蒸馏 + 策略改进"三者共享一次梯度更新；教师对学生 stop-grad。',
     size=11.5, color=INK)
footer(s, 3)
notes(s, '形式化：离线RL是带行为约束的优化；flow matching 是把策略学习变成沿概率路径的回归；'
         'FINO 把 critic、BC flow、一步蒸馏和 Q 改进合到一个损失里，一次 Adam 更新。')

# ------------------------------------------------------------------ 4 method overview
s = prs.slides.add_slide(BLANK)
header(s, '03 · 原理总览', 'FINO 的三个优化组件', 4)
mech = [
    ('① 噪声注入的 Flow Matching', '数据端注入时变噪声\n→ 扩大一步策略支持集\n（把硬约束变方差预算）',
     'x_t + η·exp(10(t−1))·ε', GREEN, GREEN_L),
    ('② 一步蒸馏', '学生 π1 逼近教师 10 步积分\n→ 推理从 10 步降到 1 步\n（α 控制保真 vs 改进）',
     'min ‖π1(s,ε) − SG[π_flow(s,ε)]‖²', BLUE, BLUE_L),
    ('③ Q-guided 采样 + 熵自适应 β', '同一状态采 K 个噪声候选\n按 softmax(βQ) 选动作\nβ 是熵约束的对偶变量',
     'K = min(10, (d+1)//2)', AMBER_D, AMBER_L),
]
cw = (CW - 0.3 * 2) / 3
for i, (t, d, formula, c, bg) in enumerate(mech):
    x = LM + i * (cw + 0.3)
    card(s, x, 1.75, cw, 3.3, fill=bg, line_color=c)
    text(s, x + 0.22, 1.9, cw - 0.4, 0.6, t, size=14.5, bold=True, color=c, line=1.2)
    text(s, x + 0.22, 2.6, cw - 0.42, 1.25, d, size=11.5, color=INK, line=1.35)
    rich(s, x + 0.22, 4.0, cw - 0.42, 0.8, [(formula, True, INK)], size=10.5, line=1.3)
card(s, LM, 5.3, CW, 1.35, fill=CARD2)
text(s, 1.0, 5.43, 11.2, 0.35, '训练/部署流水线', size=12.5, bold=True, color=NAVY)
text(s, 1.0, 5.8, 11.2, 0.7,
     '离线数据 → 训练带噪 BC flow（教师）→ 蒸馏一步策略 + 双 Q → 在线阶段：一步策略 + softmax(βQ) 采数据、'
     'β 按策略熵自适应更新。', size=11, color=INK, line=1.35)
footer(s, 4)
notes(s, '三个组件：噪声注入扩大支持集；一步蒸馏把推理从10步压到1步；Q-guided 采样和熵自适应 β 负责在线探索。'
         '整条流水线一次训练完成。')

# ------------------------------------------------------------------ 5 principles I
s = prs.slides.add_slide(BLANK)
header(s, '04 · 原理（一）', '噪声注入 = 方差正则；一步蒸馏 = 带约束的策略压缩', 5)
card(s, LM, 1.7, 5.85, 2.55, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 1.0, 1.83, 5.4, 0.35, '噪声注入（support expansion）', size=13.5, bold=True, color=GREEN)
rich(s, 1.0, 2.25, 5.4, 0.9,
     [('x_t + σ_t·ε ,   σ_t = η·exp(10(t−1)) ,  η=0.1', True, INK)], size=12.5)
bullets(s, 1.0, 2.85, 5.4, 1.3, [
    't=1（数据端）边际方差 = η² → 策略分布 = 数据 ∗ N(0, η²)',
    '等价于对速度场做输入平滑/正则，抑制高频、扩大支持集',
    '与论文 Eq.(7) 二次型 schedule 不一致（原创发现①）'], size=10.2, gap=3, bullet_color=GREEN)
card(s, 6.7, 1.7, 5.85, 2.55, fill=BLUE_L, line_color=RGBColor(0xC9, 0xD9, 0xEA))
text(s, 6.95, 1.83, 5.4, 0.35, '一步蒸馏（策略压缩）', size=13.5, bold=True, color=BLUE)
rich(s, 6.95, 2.25, 5.4, 0.9,
     [('min_θ  E[ ‖π1(s,ε) − SG[π_flow(s,ε)]‖² ]', True, INK)], size=12.5)
bullets(s, 6.95, 2.85, 5.4, 1.3, [
    'α 是"保真 vs 被 Q 拉走"的 Lagrange 权重',
    '真实 checkpoint：一步 vs 十步教师 ED≈0.006',
    '2D 玩具：ED 0.0004、推理 5.38× 加速'], size=10.2, gap=3, bullet_color=BLUE)
card(s, LM, 4.5, CW, 1.9)
text(s, 1.0, 4.63, 11.2, 0.35, '两条 schedule 的差异（原创分析素材）', size=12.5, bold=True, color=NAVY)
rich(s, 1.0, 5.05, 11.2, 0.5,
     [('代码：σ_t = 0.1·e^{10(t−1)}（t=1 最大，t=0≈0）     论文 Eq.(7)：α_t²=(η²−2η)t²+2ηt（t≈0.53 最大）',
       True, INK)], size=11.5)
text(s, 1.0, 5.6, 11.2, 0.6,
     '影响：实现的注入强度在 t<0.5 明显更小；论文公式在路径中段噪声最大。两者的"支持集扩大"行为不同 → 需要真实任务消融（§9）。',
     size=10.8, color=GRAY, line=1.3)
placeholder(s, 9.3, 4.62, 2.9, 0.5, '【待补：σ_t 曲线图】')
footer(s, 5)
notes(s, '原理一：噪声注入是方差正则/支持集扩展；一步蒸馏是带 α 权重的策略压缩。'
         '我发现代码的噪声 schedule 和论文 Eq.7 不一样，这是原创分析点。')

# ------------------------------------------------------------------ 6 principles II
s = prs.slides.add_slide(BLANK)
header(s, '04 · 原理（二）', 'Q-guided 采样与 β：熵约束的对偶上升', 6)
card(s, LM, 1.7, 6.4, 2.6)
text(s, 1.0, 1.83, 5.9, 0.35, '采样规则（在线采集）', size=13, bold=True, color=NAVY)
rich(s, 1.0, 2.25, 5.9, 1.0,
     [('K = min(10, (d+1)//2) 个噪声候选\n'
       'a = argmax 或 a ∼ softmax(β·Q/|Q̄|)', True, INK)], size=12.5, line=1.5)
bullets(s, 1.0, 2.85, 5.9, 1.3, [
    '评估用 temperature=0（argmax）；在线采集用 temperature=1',
    'd=2 时 K=1 → 机制严格退化为恒等（2D 已验证）',
    '论文任务 d=8 → K=4，机制才真正生效'], size=10.2, gap=3)
card(s, 6.95, 1.7, 5.6, 2.6, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 7.2, 1.83, 5.1, 0.35, 'β 自适应 = 对偶变量更新', size=13, bold=True, color=AMBER_D)
rich(s, 7.2, 2.25, 5.1, 1.0,
     [('β ← max(0, β + 0.1·(H + d))\n'
       'H_target = −d（目标熵）', True, INK)], size=12.5, line=1.5)
bullets(s, 7.2, 2.95, 5.1, 1.2, [
    '熵 H > H_target 时 β↑ → 更 exploitation → 熵下降（负反馈）',
    '论文 Eq.(9) 印刷符号与代码相反（原创发现②）'], size=10.2, gap=3, bullet_color=AMBER_D)
card(s, LM, 4.55, CW, 1.85, fill=CARD2)
text(s, 1.0, 4.68, 11.2, 0.35, '优化解释', size=12.5, bold=True, color=NAVY)
text(s, 1.0, 5.08, 11.2, 1.2,
     '把探索写成熵约束 min_β E[Q] + β(H(π) − H_target)：β 是约束的乘子；更新规则是对偶问题的投影梯度上升。'
     '这解释了为什么 β 只影响在线采集、不影响评估（评估用 argmax），也解释了 β 初值/目标熵的选择为何敏感。',
     size=11.3, color=INK, line=1.35)
footer(s, 6)
notes(s, '原理二：Q-guided 采样在动作维度≥4时才有意义；β 更新是对熵约束的对偶上升。'
         '论文 Eq.9 的符号和代码是反的，这是第二个原创发现。')

# ------------------------------------------------------------------ 7 implementation
s = prs.slides.add_slide(BLANK)
header(s, '05 · 实现', '实现与实验设置（与论文协议一致）', 7)
add_table(s, LM, 1.75, CW, [0.22, 0.78],
          ['项目', '设置'],
          [('平台', 'Windows 11 + WSL2/Ubuntu 24.04 + JAX 0.12(CUDA) + RTX 5070；5 处平台/依赖补丁，算法代码 0 改动'),
           ('训练预算', '1M 离线 + 500k 在线（论文原预算）；评估每 50k 步 × 50 episodes，temperature=0'),
           ('超参', 'antmaze-large：α=10、q_agg=min；humanoidmaze-medium：α=100、discount=0.995；cube-double：α=300'),
           ('数据/环境', 'OGBench 1.1.0（官方数据自动下载）；动作维 dim=8（Q-guided 机制生效）'),
           ('基线', 'FQL：同预算、同数据、同评估；唯一差异是不含噪声注入与 Q-guided/β'),
           ('计算成本', '单任务全预算 2.3–2.5 h（含评估）；共 4 条全预算运行，约 9 GPU 小时'),
           ('可复现性', '代码/补丁/脚本/CSV/日志已整理；仓库与一键命令见附录')],
          font_size=10.2, header_size=10.2, row_h=0.52)
card(s, LM, 5.85, CW, 0.85, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 1.0, 5.97, 11.2, 0.6,
     '工程要点：Windows 原生 JAX 无 CUDA 支持 → 用 WSL2 打通 GPU；官方代码只需平台/依赖适配，训练逻辑完全保留。',
     size=11.2, color=INK)
footer(s, 7)
notes(s, '实现：环境用 WSL2+RTX5070 打通 JAX GPU，只改5处平台适配；预算、评估、超参完全按论文；单任务2.3-2.5小时。')

# ------------------------------------------------------------------ 8 case study
s = prs.slides.add_slide(BLANK)
header(s, '06 · 案例研究', '案例：3 个 OGBench 任务的全预算复现', 8)
card(s, LM, 1.7, 6.0, 3.5)
text(s, 1.0, 1.83, 5.5, 0.35, '最终成功率（1.5M 步，FINO vs FQL）', size=12.5, bold=True, color=NAVY)
add_table(s, 1.0, 2.25, 5.55, [0.42, 0.16, 0.20, 0.22],
          ['任务', 'agent', '最终 success', '峰值'],
          [('antmaze-large', 'FINO', '1.00', '1.00'),
           ('antmaze-large', 'FQL', '0.92', '0.98'),
           ('humanoidmaze-medium', 'FINO', '1.00', '1.00'),
           ('cube-double-play', 'FINO', '0.72', '0.92')],
          font_size=9.6, header_size=9.2, row_h=0.36,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
text(s, 1.0, 4.15, 5.5, 0.9,
     '在线微调是关键：antmaze-large 上 FINO 从 1M 步 82% → 1.5M 步 100%；FQL 78% → 92%。',
     size=10.8, color=GREEN, line=1.3)
add_image(s, os.path.join(RES, 'fig_gpu_extended_curves.png'), 6.95, 1.7, 5.6, 3.3)
card(s, LM, 5.4, CW, 1.25, fill=CARD2)
text(s, 1.0, 5.52, 11.2, 0.35, '解读', size=12, bold=True, color=NAVY)
text(s, 1.0, 5.9, 11.2, 0.7,
     'FINO 的支持集扩展在离线期带来波动，但在线微调阶段收益显著（大概率来自噪声注入提供的可探索动作）；'
     '3 个任务均有效，其中动作维度 8 的任务 Q-guided 机制真正生效。', size=10.8, color=INK, line=1.3)
footer(s, 8)
notes(s, '案例：3个任务全预算。antmaze-large FINO 100% vs FQL 92%，在线微调从82%到100%；'
         'humanoidmaze 100%、cube-double 72%。3个任务都有效。')

# ------------------------------------------------------------------ 9 originality (complete)
s = prs.slides.add_slide(BLANK)
header(s, '07 · 原创性', '原创分析：实现审计与公式消融', 9)
card(s, LM, 1.7, 6.1, 2.75, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 1.0, 1.83, 5.6, 0.35, '① 实现审计发现（论文 vs 代码）', size=13, bold=True, color=AMBER_D)
bullets(s, 1.0, 2.2, 5.6, 2.2, [
    '噪声 schedule：代码 σ_t=0.1·e^{10(t−1)}，t=0.5 时 σ≈0.0007；论文 Eq.(7) 二次型在 t≈0.5 处 σ≈0.229（相差约 340×）',
    'β 更新：代码 β←β+0.1(H+d)；论文 Eq.(9) 印刷为 ξ−α[H−H̄]，符号方向相反',
    'to_eval 用全局 argmax：batch>1 时所有状态共享同一噪声候选',
    'CsvLogger 固定表头 / β 不随 checkpoint 保存：在线指标记录不完整'], size=8.8, gap=3, bullet_color=AMBER_D)
card(s, 6.95, 1.7, 5.63, 2.75, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 7.2, 1.83, 5.1, 0.35, '② 公式逻辑消融（真实任务，全预算）', size=13, bold=True, color=GREEN)
bullets(s, 7.2, 2.2, 5.1, 2.2, [
    'Eq.(7) 未缩放：0.04（在线崩溃）；峰值 0.229 = 官方 340×，BC flow loss 0.42',
    '峰值归一化（0.1）→ 0.94；方差读法（σ=α²）→ 1.00：形状可用，缺归一化',
    '崩溃可重复：seed1 0.12；humanoidmaze 0.00；官方 seed1 0.96',
    'Q-guided K=1/4/10：0.90/0.94/0.96；β 符号 0.94 vs 0.96（影响小）'], size=8.5, gap=2, bullet_color=GREEN)
card(s, LM, 4.7, CW, 1.7, fill=CARD2)
text(s, 1.0, 4.83, 11.2, 0.32, '③ 这些发现对"优化方法"意味着什么', size=12, bold=True, color=NAVY)
text(s, 1.0, 5.2, 11.2, 1.05,
     '这些结果把"论文公式 vs 实现"的差异变成了可复现实验：按论文给定 η 字面实现 Eq.(7) 会破坏训练（0.04/0.12/0.00），'
     '但把峰值归一化到 0.1 后恢复到 0.94–1.00 → 问题在噪声幅度的归一化，而非二次型形状；'
     'β 符号方向影响很小。对优化方法的启示：噪声方差 schedule 的关键是"幅度标定"，β 对偶步长鲁棒。', size=9.6, color=INK, line=1.35)
footer(s, 9)
notes(s, '原创性：一是论文与代码的审计差异；二是真实任务上的公式消融——论文 Eq.(7) 字面实现崩溃（0.04），'
         '官方指数 schedule 最优（1.00），beta 符号影响很小（0.94 vs 0.96）。')

# ------------------------------------------------------------------ 10 pros/cons & conclusion
s = prs.slides.add_slide(BLANK)
header(s, '08 · 结论', '优点、缺点与结论', 10)
card(s, LM, 1.7, 5.85, 2.9, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 1.0, 1.83, 5.4, 0.35, '优点', size=13.5, bold=True, color=GREEN)
bullets(s, 1.0, 2.2, 5.4, 2.3, [
    '优化稳定：监督式回归为主，避免高方差策略梯度',
    '支持集可控扩展：噪声把硬约束变成方差预算',
    '一步蒸馏：ED≈0.006、推理 5.38× 加速',
    '在线微调收益明显：antmaze 82%→100%',
    '实现简单：一次 Adam 联合更新'], size=10.2, gap=3, bullet_color=GREEN)
card(s, 6.7, 1.7, 5.85, 2.9, fill=RED_L, line_color=RGBColor(0xE8, 0xC4, 0xC4))
text(s, 6.95, 1.83, 5.4, 0.35, '缺点与风险', size=13.5, bold=True, color=RED)
bullets(s, 6.95, 2.2, 5.4, 2.3, [
    '推理成本：教师每次 batch 需 10 步前向',
    '超参敏感：α、η、β 都需按任务调',
    '单步蒸馏误差（高维任务 ED 更大）',
    '理论缺口：schedule 与 β 符号与论文不一致',
    '实验覆盖有限：单 seed、3/45 任务'], size=10.2, gap=3, bullet_color=RED)
card(s, LM, 4.8, CW, 1.55, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
text(s, 1.0, 4.93, 11.2, 0.35, '结论（一句话）', size=12.5, bold=True, color=AMBER_D)
text(s, 1.0, 5.33, 11.2, 0.9,
     'FINO 把离线到在线的策略优化转化为"噪声正则的生成式回归 + 蒸馏 + 对偶式探索"；复现（3 任务，antmaze 100% vs FQL 92%）'
     '验证其优化有效，原创审计与消融揭示了实现与论文的偏差及机制边界。', size=11.5, color=INK, line=1.35)
text(s, 1.0, 6.6, 11.2, 0.35, '谢谢 / Q&A · 代码与数据：github.com/engulf408/FINO', size=11, color=GRAY)
footer(s, 10)
notes(s, '结论：FINO 是把策略优化变成"噪声正则回归+蒸馏+对偶探索"的新优化方法；'
         '复现有效、审计发现偏差、消融设计用于验证机制。谢谢。')

# ---------------------------------------------------------------- save
os.makedirs(OUT_DIR, exist_ok=True)
prs.core_properties.title = 'FINO：面向离线到在线强化学习的生成式策略优化方法（10 分钟汇报骨架）'
prs.core_properties.author = '＿＿＿＿'
prs.save(OUT)
print('saved:', OUT, '| slides:', len(prs.slides._sldIdLst))
