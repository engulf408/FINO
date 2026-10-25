# -*- coding: utf-8 -*-
"""Compact 5-minute FINO reproduction deck: how / how much / conclusions.

Output: E:\\FINO\\ppt\\FINO复现-5分钟汇报.pptx
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

OUT = r'E:\FINO\ppt\FINO复现-5分钟汇报.pptx'
RES = r'E:\FINO\results'

NAVY = RGBColor(0x0F, 0x1B, 0x2D)
INK = RGBColor(0x1B, 0x26, 0x3B)
GRAY = RGBColor(0x55, 0x61, 0x76)
LGRAY = RGBColor(0x8B, 0x96, 0xA8)
LINE = RGBColor(0xE2, 0xE7, 0xEE)
CARD = RGBColor(0xF5, 0xF7, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xF2, 0xA3, 0x3A)
AMBER_D = RGBColor(0xC9, 0x7F, 0x1C)
AMBER_L = RGBColor(0xFD, 0xF2, 0xE2)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
GREEN_L = RGBColor(0xE9, 0xF5, 0xEE)
BLUE = RGBColor(0x3B, 0x6F, 0xA0)
BLUE_L = RGBColor(0xE8, 0xEF, 0xF7)
RED = RGBColor(0xC0, 0x4B, 0x4B)
FONT = 'Microsoft YaHei'
SW, SH = 13.333, 7.5
LM, CW = 0.8, 11.73


def set_font(run, size=12, bold=False, color=INK, spc=None):
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


def text(slide, x, y, w, h, s, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, line=1.15, spc=None):
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


def rich(slide, x, y, w, h, parts, size=14, color=INK, align=PP_ALIGN.LEFT, line=1.15):
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


def card(slide, x, y, w, h, fill=CARD, line_color=LINE):
    return rect(slide, x, y, w, h, fill=fill, line_color=line_color,
                shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)


def header(slide, tag, title, num):
    rect(slide, 0, 0, SW, 0.12, fill=NAVY)
    text(slide, LM, 0.4, 9, 0.3, tag, size=11, bold=True, color=AMBER_D, spc=1.5)
    text(slide, LM, 0.7, 11.7, 0.75, title, size=30, bold=True, color=NAVY)
    rect(slide, LM, 1.52, 0.6, 0.05, fill=AMBER)
    rect(slide, LM + 0.7, 1.54, CW - 0.7, 0.012, fill=LINE)
    text(slide, SW - 1.05, 0.4, 0.5, 0.3, f'{num:02d}', size=11, bold=True, color=LGRAY, align=PP_ALIGN.RIGHT)


def footer(slide, num):
    rect(slide, LM, 7.0, CW, 0.01, fill=RGBColor(0xE7, 0xEB, 0xF2))
    text(slide, LM, 7.08, 8, 0.25, 'FINO 复现 · ICLR 2026', size=9, color=LGRAY)
    text(slide, SW - 1.2, 7.08, 0.5, 0.25, str(num), size=10, bold=True, color=LGRAY, align=PP_ALIGN.RIGHT)


def add_image(slide, path, x, y, max_w, max_h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    left, top = x + (max_w - w) / 2, y + (max_h - h) / 2
    slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(w), Inches(h))
    return left, top, w, h


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]

# ============================================================ 1 cover
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, fill=NAVY)
rect(s, 0.95, 1.15, 0.14, 0.14, fill=AMBER)
text(s, 1.2, 1.08, 9, 0.3, 'ICLR 2026 · OFFICIAL IMPLEMENTATION REPRODUCTION',
     size=11, bold=True, color=RGBColor(0x9F, 0xB0, 0xC8), spc=1.8)
text(s, 0.95, 1.6, 11.5, 1.0, 'FINO 复现：怎么做的、做了多少、结论', size=42, bold=True, color=WHITE)
text(s, 0.95, 2.75, 10.5, 0.5,
     'Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning',
     size=17, color=RGBColor(0xC3, 0xCF, 0xE0))
rect(s, 0.95, 3.5, 1.2, 0.04, fill=AMBER)
text(s, 0.95, 3.85, 11.3, 0.6,
     '核心结果：3 个 OGBench 任务按论文原预算复现 —— antmaze-large 100%、humanoidmaze-medium 100%、'
     'cube-double-play 72%（FQL 对比 92%）。',
     size=18, bold=True, color=WHITE, line=1.35)
text(s, 0.95, 5.0, 11.0, 0.4, '5 分钟汇报 · 2026-10-21 · github.com/engulf408/FINO',
     size=12, color=RGBColor(0x8F, 0xA6, 0xC2))
text(s, 0.95, 6.5, 11.0, 0.3, 'Windows + WSL2 + RTX 5070 · JAX 0.12（CUDA）· OGBench 1.1.0',
     size=10, color=RGBColor(0x6F, 0x82, 0x9C))
notes(s, '开场一句话：我用 GPU 环境把官方 FINO 按论文预算复现了 3 个任务，antmaze-large 上 FINO 100%、FQL 92%。'
         '下面用三页说清：怎么做的、做了多少、结论是什么。')

# ============================================================ 2 method
s = prs.slides.add_slide(BLANK)
header(s, '01 · 怎么复现', '三步：跑通环境 → 按论文协议训练 → 与 FQL 对比', 2)
steps = [
    ('① 环境', '官方代码在 Windows 只能 CPU（JAX CUDA 无 Windows 版）→ 装 WSL2 + Ubuntu，用 RTX 5070 跑 JAX GPU。'
              '共 5 处平台/依赖补丁，算法代码 0 改动。'),
    ('② 设置', 'OGBench 官方数据；严格按论文预算：1M 离线 + 500k 在线，每 50k 步评估 50 episodes；'
              '超参按 README（alpha、q_agg、discount）。'),
    ('③ 运行', '每个任务 FINO 与 FQL 同预算对比；单任务约 2.3–2.5 小时（GPU，含评估）；'
              '共 4 条全预算运行、约 9 小时 GPU。'),
]
cw = (CW - 0.3 * 2) / 3
for i, (t, d) in enumerate(steps):
    x = LM + i * (cw + 0.3)
    card(s, x, 1.9, cw, 3.15, fill=WHITE)
    rect(s, x, 1.9, cw, 0.09, fill=AMBER)
    text(s, x + 0.25, 2.15, cw - 0.5, 0.4, t, size=19, bold=True, color=NAVY)
    text(s, x + 0.25, 2.7, cw - 0.5, 2.2, d, size=13.5, color=INK, line=1.4)
rich(s, LM, 5.35, CW, 0.5, [('一句话：', True, AMBER_D),
                           ('除了平台/依赖适配，训练流程、损失函数、采样逻辑完全使用官方代码。', False, INK)], size=16)
text(s, LM, 6.0, CW, 0.9,
     '可复现：安装脚本（wsl_setup/）、训练命令、日志与 CSV 全部保留在仓库中；同一命令可在任意 Linux + GPU 机器重跑。',
     size=12, color=GRAY, line=1.3)
footer(s, 2)
notes(s, '怎么复现：三步。环境上，Windows 原生 JAX 用不了 GPU，所以装 WSL2 + Ubuntu，让 RTX 5070 跑 JAX；'
         '代码只改了 5 处平台/依赖适配，算法没动。设置完全按论文预算和 README 超参。运行上 FINO 与 FQL 同预算对比，'
         '单任务约 2.3-2.5 小时。')

# ============================================================ 3 coverage
s = prs.slides.add_slide(BLANK)
header(s, '02 · 复现了多少', '论文主表 45 个任务，完整复现 3 个（≈7%）', 3)
kpis = [
    ('3 / 45', '任务全预算复现', GREEN),
    ('4 条', '全预算运行（FINO×3 + FQL×1）', AMBER),
    ('3 个', '核心机制 2D 定量验证', BLUE),
    ('0', '消融 / 多 seed / D4RL', RED),
]
kw = (CW - 0.25 * 3) / 4
for i, (v, l, c) in enumerate(kpis):
    x = LM + i * (kw + 0.25)
    card(s, x, 1.9, kw, 1.35)
    rect(s, x, 2.05, 0.06, 1.05, fill=c)
    text(s, x + 0.22, 2.0, kw - 0.35, 0.55, v, size=28, bold=True, color=NAVY)
    text(s, x + 0.22, 2.62, kw - 0.35, 0.5, l, size=11, color=GRAY, line=1.1)
card(s, 0.8, 3.65, 5.85, 2.4, fill=GREEN_L, line_color=RGBColor(0xC9, 0xE4, 0xD3))
text(s, 1.05, 3.8, 5.3, 0.35, '已完成（全预算、论文协议）', size=15, bold=True, color=GREEN)
text(s, 1.05, 4.25, 5.3, 1.7,
     '• antmaze-large：FINO 100% vs FQL 92%\n'
     '• humanoidmaze-medium：FINO 100%\n'
     '• cube-double-play：FINO 72%（峰值 92%）\n'
     '• 3 个核心机制 2D 定量验证：噪声注入、一步蒸馏、Q-guided+β',
     size=12.5, color=INK, line=1.45)
card(s, 6.85, 3.65, 5.68, 2.4, fill=RGBColor(0xFB, 0xEE, 0xEE), line_color=RGBColor(0xE8, 0xC4, 0xC4))
text(s, 7.1, 3.8, 5.2, 0.35, '未覆盖（如实说明）', size=15, bold=True, color=RED)
text(s, 7.1, 4.25, 5.2, 1.7,
     '• 其余 42 个任务（含 D4RL 10 个）\n'
     '• 论文真实环境消融实验\n'
     '• 多 seed 方差\n'
     '• 像素环境 / 视频',
     size=12.5, color=INK, line=1.45)
text(s, LM, 6.35, CW, 0.4,
     '口径：45 = OGBench 35（7 环境 × 5 task）+ D4RL 10；本复现只覆盖 OGBench 的 3 个任务。',
     size=11.5, color=GRAY)
footer(s, 3)
notes(s, '做了多少：论文 README 一共 45 个任务命令，我们完整按论文预算复现了 3 个 OGBench 任务，约占 7%；'
         '另外用 2D 玩具实验定量验证了三个核心机制。没有做：其余 42 个任务（含 D4RL 10 个）、真实环境消融、多 seed、像素环境。')

# ============================================================ 4 results
s = prs.slides.add_slide(BLANK)
header(s, '03 · 结果', '关键结果：3 个任务均学到有效策略', 4)
card(s, 0.8, 1.85, 6.15, 3.5)
text(s, 1.05, 2.0, 5.6, 0.35, '全预算最终成功率（1.5M 步）', size=15, bold=True, color=NAVY)
rows = [
    ('antmaze-large', 'FINO', '100%', GREEN),
    ('antmaze-large', 'FQL', '92%', BLUE),
    ('humanoidmaze-medium', 'FINO', '100%', GREEN),
    ('cube-double-play', 'FINO', '72%', AMBER_D),
]
yy = 2.55
for task, agent, val, c in rows:
    text(s, 1.05, yy, 2.9, 0.35, task, size=13, color=INK)
    text(s, 3.9, yy, 0.7, 0.35, agent, size=11.5, color=GRAY)
    text(s, 4.9, yy - 0.04, 1.55, 0.45, val, size=22, bold=True, color=c, align=PP_ALIGN.RIGHT)
    rect(s, 1.05, yy + 0.45, 5.55, 0.01, fill=LINE)
    yy += 0.65
add_image(s, os.path.join(RES, 'fig_gpu_task_summary.png'), 7.1, 1.85, 5.5, 3.55)
card(s, 0.8, 5.55, 11.73, 1.15, fill=AMBER_L, line_color=RGBColor(0xF3, 0xD8, 0xB0))
rich(s, 1.05, 5.72, 11.2, 0.35, [('在线微调是关键：', True, AMBER_D),
                                 ('antmaze-large 上 FINO 从 1M 步的 82% 提升到 1.5M 步的 100%；FQL 从 78% 到 92%。', False, INK)], size=14)
text(s, 1.05, 6.15, 11.2, 0.35,
     '单 seed（seed=0）；humanoidmaze 评估为每 100k×20 episodes，其余为每 50k×50 episodes。',
     size=10.5, color=GRAY)
footer(s, 4)
notes(s, '结果：三个任务全预算，antmaze-large FINO 100%、FQL 92%；humanoidmaze 100%；cube-double 72%，峰值 92%。'
         '最重要的是在线微调阶段把 FINO 从 82% 拉到 100%，与论文主张一致。')

# ============================================================ 5 conclusions
s = prs.slides.add_slide(BLANK)
header(s, '04 · 结论', '三句话总结', 5)
concl = [
    ('① 实现与机制成立', '官方代码全链路跑通（训练/在线微调/评估/存档恢复）；三个核心机制在真实任务与 2D 实验中均按论文描述工作；'
                    '蒸馏保真（一步 vs 10 步教师 ED 0.006）。', GREEN, GREEN_L),
    ('② 论文主表只覆盖约 7%', '45 个任务中完成 3 个全预算（≈7%）；数值上与论文同任务量级（antmaze-large 100%），'
                          '但没有 D4RL、真实环境消融、多 seed，不能声称全量复现。', AMBER_D, AMBER_L),
    ('③ 全量复现的成本', '再补 3–5 个代表任务 + 多 seed + 消融约 50–100 GPU 小时；'
                           '全量 45 任务约 400–900 GPU 小时（本机约 20–40 天）。', BLUE, BLUE_L),
]
for i, (t, d, c, bg) in enumerate(concl):
    y = 1.9 + i * 1.55
    card(s, 0.8, y, 11.73, 1.35, fill=bg, line_color=c)
    text(s, 1.1, y + 0.15, 4.0, 0.4, t, size=18, bold=True, color=c)
    text(s, 4.4, y + 0.13, 7.9, 1.1, d, size=12.5, color=INK, line=1.3)
text(s, 0.8, 6.65, 11.7, 0.3, '仓库与报告：github.com/engulf408/FINO · docs/02_reproduction_report.md',
     size=11, color=GRAY)
footer(s, 5)
notes(s, '结论三句话：第一，实现和机制是成立的，跑通且有定量验证；第二，论文主表只覆盖约 7%，不能声称全量复现；'
         '第三，如果要做全量，大约需要 400-900 GPU 小时。')

# ============================================================ 6 thanks
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, fill=NAVY)
rect(s, 0.95, 1.7, 1.2, 0.04, fill=AMBER)
text(s, 0.95, 2.15, 8.5, 1.2, '谢谢', size=54, bold=True, color=WHITE)
text(s, 0.95, 3.45, 8.5, 0.4, 'Q & A', size=18, color=RGBColor(0x9F, 0xB0, 0xC8), spc=2)
text(s, 0.95, 4.35, 11.2, 0.4, 'github.com/engulf408/FINO', size=16, bold=True, color=WHITE)
text(s, 0.95, 4.85, 11.2, 0.4, '复现报告：docs/02_reproduction_report.md · 汇报 PPT：ppt/FINO复现-5分钟汇报.pptx',
     size=12.5, color=RGBColor(0x8F, 0xA6, 0xC2))
text(s, 0.95, 6.6, 11.2, 0.3, 'FINO 复现 · 2026-10-21', size=10, color=RGBColor(0x6F, 0x82, 0x9C))
notes(s, '结束，欢迎提问。')

prs.core_properties.title = 'FINO 复现：怎么做的、做了多少、结论'
prs.core_properties.author = 'engulf408'
prs.save(OUT)
print('saved:', OUT, '| slides:', len(prs.slides._sldIdLst))
