# -*- coding: utf-8 -*-
"""Update the reproduction report with the full-budget GPU results (2026-10-21)."""
import io

P = r'E:\FINO\docs\02_reproduction_report.md'
s = io.open(P, encoding='utf-8').read()


def rep(old, new, count=1):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:80]
    s = s.replace(old, new, count)


# ---------------- header ----------------
rep('> 报告日期：2026-10-20\n'
    '> 配套材料：`docs/01_code_analysis.md`（源码逐行分析）、`docs/patches.diff`（为跑通而做的 4 处最小改动）、'
    '`experiments/toy/RESULTS.md`（2D 机制验证）、`results/`（曲线与汇总表）、`logs/`（全部运行日志）',
    '> 报告日期：2026-10-20（初版）/ **2026-10-21 更新：WSL2 + RTX 5070 全预算 GPU 复现结果**\n'
    '> 配套材料：`docs/01_code_analysis.md`（源码逐行分析）、`docs/patches.diff`（平台适配补丁）、'
    '`experiments/toy/RESULTS.md`（2D 机制验证）、`results/`（CPU/GPU 曲线与汇总表）、'
    '`gpu_exp/`（全预算 GPU 运行 CSV/checkpoint）、`wsl_setup/`（WSL2 环境脚本）、`logs/`（全部运行日志）')

# ---------------- §0 ----------------
rep('2. **无法在本机做"论文数值级"复现**：JAX 的 CUDA 插件只有 Linux 轮子，本机 Windows 无 WSL/Docker/管理员权限，'
    '**RTX 5070 无法被 JAX 使用**，只能 CPU。实测 antmaze-large 约 **15.5 update/s（64 ms/step）**，'
    '论文单任务 1.5M 步 ≈ **26.7 小时**，50+ 任务 ≈ **55 天**（单进程），不可行。',
    '2. **GPU 环境已打通，并完成论文级全预算复现（2026-10-21 更新）**：用户授权管理员权限后安装 WSL2 + Ubuntu 24.04，'
    'JAX 0.12.0(CUDA) 在 RTX 5070 上可用；按论文预算（1M 离线 + 500k 在线、每 50k×50 episodes）复现 antmaze-large，'
    '**FINO 最终成功率 100%（1.00），FQL 92%（0.92）**，单任务 2.5/2.3 小时。'
    '此前 CPU 阶段的限制与 3% 预算运行保留在 §2/§7.2b 作为历史记录。')
rep('3. **交付的是"代码/方法级复现 + 小预算部分复现"**：',
    '3. **交付内容**（CPU 阶段 + GPU 阶段）：')
rep('   - 论文任务 antmaze-large-navigate-singletask-v0 做了 30k 离线 + 10k 在线的缩减预算实验（约 3% 论文预算），'
    '曲线与失败/成功情况如实报告；',
    '    - 论文任务 antmaze-large 先做 3% 预算 CPU 运行（§7.2b），后完成 **全预算 GPU 运行**（§7.2，核心结果）；'
    '扩展任务 humanoidmaze-medium / cube-double-play 正在 GPU 上运行；')
rep('4. **发现并修正了 4 个"非 Linux/非旧版依赖"环境下会直接崩溃的问题**',
    '4. **发现并修正了 5 个平台/依赖兼容问题**（4 个 Windows/CPU + 1 个 GPU 栈 distrax 兼容）')

# ---------------- §1 ----------------
rep('3. 论文任务的部分复现：在 antmaze-large 上做预算缩减实验（§6.2、§7.2）。\n'
    '4. 机制验证与复现成本核算：2D 玩具实验 + CPU 计时外推，给出在 Linux+GPU 上全量复现的可行方案（§7.3、§7.4、§9）。',
    '3. 论文任务的完整复现：antmaze-large 全预算（1M+500k）FINO vs FQL（§7.2），并扩展 humanoidmaze-medium / cube-double-play；\n'
    '4. 机制验证与成本分析：2D 玩具实验（§7.3）+ CPU/GPU 计时对比（§7.4）+ 可复刻的 WSL2 GPU 环境（§2.3、§9）。')
rep('**不在范围内**：D4RL 系列（`d4rl`/`mujoco-py`/`gym==0.23.1` 在 Windows 上基本不可安装）、'
    '像素环境（`impala_small` 编码器/视频渲染）、全部 50+ 任务的完整训练。',
    '**不在范围内**：D4RL 系列 10 个任务（需要 `d4rl`/`mujoco-py`/MuJoCo 2.1）、像素环境（`impala_small` 编码器/视频渲染）、'
    'README 全部 55 个任务的完整训练（本机 GPU 单卡约需 5–9 天）。')

# ---------------- §2 table + §2.3 ----------------
rep('| GPU | NVIDIA GeForce RTX 5070 12 GB（驱动 617.14，CUDA UMD 13.4）——**JAX 用不到** |\n'
    '| 系统级虚拟化 | WSL 未安装；无 Docker、无管理员权限（`IsInRole(Administrator)=False`） |',
    '| GPU | NVIDIA GeForce RTX 5070 12 GB（驱动 617.14，CUDA UMD 13.4）——已通过 WSL2 用于 JAX |\n'
    '| 系统级虚拟化 | WSL2 + Ubuntu 24.04.3（E 盘 VHDX）；初版时无管理员权限，2026-10-21 由用户授权后完成安装 |')

anchor = '| antmaze 环境 step | 0.26 ms |\n'
gpu_section = '''
### 2.3 GPU 环境（WSL2 + RTX 5070，2026-10-21 补充）

CPU 阶段的限制已于 2026-10-21 解决：用户以管理员身份重启 DSH 授权后，安装了 WSL2 + Ubuntu 24.04.3，并用 RTX 5070 运行 JAX CUDA。

安装要点（脚本见 `wsl_setup/`、日志见 `logs/`）：

1. 启用 Windows 功能 `Microsoft-Windows-Subsystem-Linux`、`VirtualMachinePlatform` → 重启；
2. `winget install Microsoft.WSL`（2.7.13），下载 Ubuntu 24.04 base rootfs 后执行
   `wsl --import FINO-Ubuntu E:\\FINO\\wsl\\FINO-Ubuntu <rootfs> --version 2`（VHDX 放 E 盘，不占 C 盘）；
3. apt 使用清华镜像（最小 rootfs 先用 HTTP 引导安装 `ca-certificates`，再切回 HTTPS）；
4. `/opt/fino-venv`：`jax 0.12.0 + CUDA 插件`、`flax 0.12.10`、`optax 0.2.8`、`orbax-checkpoint 0.12.8`、
   `ogbench`、`mujoco 3.3.0`、`gymnasium 0.29.1` 等（清华 PyPI 镜像）；
5. 代码适配只有 1 处：`utils/networks.py` 的 distrax 导入改为可选（distrax 0.1.5 与 jax 0.12 不兼容；
   FINO/FQL 不使用高斯 Actor；新增补丁 #5）；
6. 安装 `libegl1 libgl1 libosmesa6` 等库满足 mujoco 导入期对 EGL 库的依赖（训练/评估不创建 GL 上下文）。

GPU 实测（RTX 5070 12 GB，`jax.devices() = [CudaDevice(id=0)]`）：

| 场景 | 实测 |
|---|---|
| FINO update（antmaze，batch=256） | **1.58 ms**（基准）；训练循环约 2.4 ms/步 |
| FQL update（同规模） | 1.65 ms |
| 数据加载（antmaze-large，100 万条） | 1.3 s |
| 环境 step（antmaze） | 0.21 ms |
| 全预算单任务（1.5M 步 + 31 次评估） | **2.3–2.5 小时** |

因此 55 个任务在本机 GPU 上约需 5–9 天（单卡）；3–5 个代表任务可在 1–2 天内完成。
'''
rep(anchor, anchor + gpu_section)

# ---------------- §5 ----------------
rep('## 5. Windows / CPU 适配补丁（4 处，最小侵入）', '## 5. 平台适配补丁（5 处，最小侵入）')
rep('| 4 | `utils/evaluation.py:60` | `np.random.randint(0, 2**32)` → `np.random.randint(0, 2**31-1)` | '
    'Windows 上 numpy 默认整型为 int32，上界 2^32 越界（`ValueError: high is out of bounds for int32`） |',
    '| 4 | `utils/evaluation.py:60` | `np.random.randint(0, 2**32)` → `np.random.randint(0, 2**31-1)` | '
    'Windows 上 numpy 默认整型为 int32，上界 2^32 越界（`ValueError: high is out of bounds for int32`） |\n'
    '| 5 | `utils/networks.py:3` | `import distrax` 改为 try/except 可选导入（WSL/GPU 栈） | '
    'distrax 0.1.5 与 jax 0.12 不兼容（`jax.interpreters.xla.pytype_aval_mappings` 已移除）；FINO/FQL 不使用高斯 Actor |')
rep('四处都不改变算法/训练逻辑；除第 3 处外，在 Linux 上也不会破坏原行为（第 2 处可用环境变量 `FINO_WANDB_MODE=online` 恢复）。',
    '5 处都不改变算法/训练逻辑：4 处在 Windows/CPU 环境下生效，第 5 处仅在新版 JAX 的 GPU 环境下生效；'
    '在 Linux 上第 2 处可用环境变量 `FINO_WANDB_MODE=online` 恢复。')

# ---------------- §6.2 ----------------
rep('| （可选）baseline `antmaze-large`（FQL） | 30,000 | 10,000 | 同 FINO | 验证 `--agent=agents/fql.py` 切换与同预算对比 |\n\n'
    '评估 episodes 与间隔都做了缩减（论文为每 50k 步 × 50 episodes），这是纯工程性缩减，不改变算法。',
    '| （可选）baseline `antmaze-large`（FQL） | 30,000 | 10,000 | 同 FINO | 验证 `--agent=agents/fql.py` 切换与同预算对比 |\n'
    '| **全预算 GPU** antmaze-large（FINO） | **1,000,000** | **500,000** | 每 50k 步 × 50 episodes | 论文原设置（WSL2 + RTX 5070） |\n'
    '| **全预算 GPU** antmaze-large（FQL） | **1,000,000** | **500,000** | 同上 | 同预算 baseline（论文原设置） |\n'
    '| 扩展 GPU humanoidmaze-medium / cube-double-play（FINO） | **1,000,000** | **500,000** | 同上 | 扩展任务（α=100 / 300，discount 按 README） |\n\n'
    'CPU 阶段的两条运行使用了缩减预算与评估；GPU 阶段的两条主运行**完全采用论文预算与评估协议**，可直接与论文表格对比。')

# ---------------- §7.2 ----------------
old_head = '### 7.2 论文任务 antmaze-large-navigate-singletask-v0（缩减预算）\n\n'
new_head = '''### 7.2 论文任务 antmaze-large-navigate-singletask-v0（全预算 GPU 复现，核心结果）

> **结果概览：FINO 最终成功率 100%（1.00），FQL 92%（0.92）**；每 50k 步评估 50 episodes，共 31 个评估点；
> FINO 在线微调阶段成功率从 1M 步的 0.82 提升到 1.5M 步的 1.00，FQL 从 0.78 到 0.92。

**运行设置（与论文一致）**：1M 离线 + 500k 在线；超参按 README（`--agent.alpha=10 --agent.q_agg=min`）；
每 50k 步评估 50 episodes（`temperature=0` argmax）；seed=0；环境 WSL2 + Ubuntu 24.04 + JAX 0.12.0(CUDA) + RTX 5070。

![antmaze-large 全预算 FINO vs FQL 评估曲线](../results/fig_gpu_eval_antmaze-large-navigate-singletask-v0.png)

| 指标 | FINO | FQL |
|---|---:|---:|
| **最终 success（1.5M 步）** | **1.00** | 0.92 |
| 峰值 success | 1.00 | 0.98 |
| 1M 步（离线结束）success | 0.82 | 0.78 |
| 最终 episode return | **−383.4** | −517.9 |
| 总耗时 / 平均速度\\* | 150.3 min / 6.01 ms/step | 136.6 min / 5.47 ms/step |
| 训练损失终点（BC flow / critic） | 0.2846 / 0.2116 | 0.2790 / 0.0924 |
| 策略动作 Q 终点 | −90.69 | −92.64 |

\\* 含 31 次评估（每次 50 episodes × 1000 步）；纯训练 update 基准为 FINO 1.58 ms / FQL 1.65 ms（§2.3）。

**结论**：

1. **论文级效果已复现**：FINO 全预算达到 **100%** 成功率（FQL 92%），在线微调阶段成功率明显上升（0.82→1.00），
   与论文"离线预训练 + 在线微调"的设定一致；这是本复现最关键的正面结果。
2. FINO 最终比 FQL 高 8 个百分点、return 高约 26%（−383 vs −518），方向与论文一致；但为 **单 seed**，
   不能作显著性结论（见 §8.1）。
3. 单任务全预算约 **2.3–2.5 小时**（RTX 5070，含评估），比 CPU 版 26.7 小时快约 10–11 倍；纯训练快约 26–40 倍。

**全预算 checkpoint 级分析**（1M 步 checkpoint，256 个验证状态；`results/checkpoint_analysis_full_budget.json`）：

| 指标 | FINO | FQL | 说明 |
|---|---:|---:|---|
| 一步策略 vs 10 步教师 ED | 0.0061 | 0.0055 | 蒸馏保真（与 30k 步时 0.0078 同量级） |
| 一步策略 vs 数据集动作 MSE | 0.0901 | 0.0904 | 比 30k 步时（0.118）明显下降 |
| 10 步教师 vs 数据集动作 MSE | 0.0595 | 0.0602 | 比 30k 步时（0.077）下降，拟合更充分 |
| 贪心策略平均 Q | −92.1 | −92.0 | 接近理论下限 −100 |

#### 7.2b 历史参照：3% 预算 CPU 运行（2026-10-20）

'''
rep(old_head, new_head)

# ---------------- §7.4 ----------------
old74 = '''| 项目 | 实测/外推 |
|---|---|
| FINO update（antmaze 规模） | 64 ms（15.5 it/s） |
| 论文单任务（1.5M 步） | ≈ 26.7 h（本机 CPU） |
| OGBench 默认 7 任务 | ≈ 7.8 天 |
| README 全量 45（OGBench）+10（D4RL）任务 | ≈ 55 天 |
| 8 进程并行（8 核） | 每进程约 1 核，合计吞吐提升有限，仍约 1 周以上 |'''
new74 = '''| 项目 | CPU（Windows） | GPU（WSL2 + RTX 5070） |
|---|---|---|---|
| FINO update（antmaze 规模） | 64 ms（15.5 it/s） | **1.58 ms**（约 40×） |
| 全预算单任务（1.5M 步 + 31 次评估） | ≈ 26.7 h | **2.3–2.5 h**（实测） |
| OGBench 默认 7 任务 | ≈ 7.8 天 | **约 17–20 小时** |
| README 全量 45（OGBench）+10（D4RL） | ≈ 55 天 | **5–9 天**（单卡） |'''
rep(old74, new74)

# ---------------- §8.1 ----------------
old81 = '''1. **预算**：离线/在线步数缩减到论文的 0.4%~3%，评估 episodes 与间隔缩减；这直接导致成功率为 0，**不能**与论文表格数值比较。
2. **硬件**：CPU 而非 GPU；浮点运算顺序/精度差异（XLA CPU vs CUDA）会造成微小差异，但对 512×512 MLP 影响可忽略。
3. **超参**：沿用 README 的 `alpha`、`q_agg`、`discount`，未重新调参；`beta` 初值/更新率按默认。
4. **环境/数据版本**：ogbench 1.1.0（README 钉的版本），但本机 Python 3.11/jax 0.4.35 与原作者的 Python 3.9/jax 版本不同；OGBench 文档提到 1.2.0 起 singletask 的 reward/terminated 计算时点由 `s'` 改为 `s`，1.1.0 不受影响，但与"最新版本"复现者会有差异。
5. **D4RL 未覆盖**：antmaze-umaze 等 10 个 D4RL 任务需要 `d4rl + mujoco-py + gym==0.23.1`，Windows 无法构建。
6. **wandb 离线**：无云端曲线，所有指标以 CSV 形式保存（`exp/FINO/<env>/0/*.csv`）。
7. **视频渲染未验证**：`video_episodes=0`；Windows 需设 `MUJOCO_GL=wgl/glfw`，且 `utils/log_utils.py:get_wandb_video` 还依赖 moviepy/wandb 视频链路。'''
new81 = '''1. **运行预算**：antmaze-large 的两条主运行**完全按论文预算**（1M + 500k）；§7.2b 的 CPU 运行是 3% 预算的历史记录，仅作链路验证。
2. **硬件**：RTX 5070 12 GB（消费级）；论文环境应为更大显存的 GPU，但本任务 MLP 规模小，显存与算力足够；CUDA 与 CPU 的浮点差异可忽略。
3. **软件版本**：为支持 Blackwell（sm_120）使用了 jax 0.12.0 / flax 0.12.10 / orbax 0.12.8（官方 requirements 为 Python 3.9 + jax≥0.4.26）；官方代码仅需 1 处 distrax 可选导入适配，算法与损失未改。
4. **超参**：严格按 README（antmaze-large `alpha=10`、`q_agg=min`；其余默认）；未重新调参。
5. **评估协议**：与论文一致（每 50k 步 × 50 episodes、temperature=0 argmax），因此成功率可直接对比。
6. **单 seed**：所有运行 seed=0，未做多 seed 平均；FINO 100% vs FQL 92% 的差距方向可信，但统计显著性未检验。
7. **D4RL 未覆盖**：10 个 D4RL 任务需要 `d4rl + mujoco-py` + MuJoCo 2.1 二进制，本次未安装（Linux 下可补）。
8. **wandb 离线 + 无视频**：所有指标从 CSV 重建；`video_episodes=0`，视频链路未验证。
9. **数据版本**：ogbench 1.1.0（与 README 一致）；singletask reward/terminated 计算时点与 1.2.0+ 有已知差异（见 OGBench 文档）。'''
rep(old81, new81)

io.open(P, 'w', encoding='utf-8').write(s)
print('report patched ok, lines:', len(s.splitlines()))
