# FINO 复现工作区（Windows + CPU / WSL2 + GPU）

本目录是对 <https://github.com/CTID282/FINO>（ICLR 2026 论文 *Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning* 官方实现）的复现工作区。

## 最新结果（2026-10-21，GPU 全预算）

> 在 WSL2 + RTX 5070 + JAX 0.12(CUDA) 上按**论文原预算**（1M 离线 + 500k 在线、每 50k 步评估 50 episodes）完成 antmaze-large 复现：
>
> **FINO 最终成功率 100%（1.00），FQL 92%（0.92）**；FINO 在线微调阶段 0.82 → 1.00；单任务 2.3–2.5 小时。
>
> 详细数据：[`results/gpu_summary.md`](results/gpu_summary.md)、[`gpu_exp/`](gpu_exp/)、[`docs/02_reproduction_report.md`](docs/02_reproduction_report.md) §7.2。

## 先看这里

| 想了解 | 打开 |
|---|---|
| 复现报告（环境、补丁、CPU+GPU 实验、结果、偏差、复现路径） | [`docs/02_reproduction_report.md`](docs/02_reproduction_report.md) |
| 源码/方法逐行技术分析（724 行） | [`docs/01_code_analysis.md`](docs/01_code_analysis.md) |
| 平台适配补丁（4 处 Windows/CPU + 1 处 GPU） | [`docs/patches.diff`](docs/patches.diff) |
| WSL2 + GPU 环境（安装/运行脚本） | [`wsl_setup/`](wsl_setup/) |
| 2D 玩具机制验证报告 | [`experiments/toy/RESULTS.md`](experiments/toy/RESULTS.md) |
| 结果曲线与汇总 | [`results/`](results/) |
| 全部原始日志 | [`logs/`](logs/) |
| 汇报 PPT | [`ppt/FINO复现工作与结果分析.pptx`](ppt/FINO复现工作与结果分析.pptx) |

## 目录

```
FINO/
├── FINO/                  # 官方源码 + 5 处平台适配补丁（算法未改）
├── _extract/FINO-main/    # 未改动的官方源码备份（用于 diff）
├── .venv/                 # Windows CPU 环境（Python 3.11 + JAX 0.4.35 CPU）
├── wsl_setup/             # WSL2 + Ubuntu + JAX CUDA 安装脚本
├── constraints.txt        # Windows CPU 环境钉版依赖
├── docs/                  # 01 源码分析、02 复现报告、patches.diff、requirements-lock.txt
├── scripts/               # 复现运行/下载/绘图/GPU 基准与结果分析脚本
│   ├── bench_gpu.py               # GPU 速度基准（FINO/FQL/环境）
│   ├── analyze_gpu_results.py     # 全预算结果汇总与曲线
│   ├── analyze_ckpt_full.py       # 全预算 checkpoint 级分析
│   ├── analyze_checkpoint.py      # 小预算 checkpoint 分析
│   ├── run_repro.ps1              # 一键运行 CPU 阶段实验
│   ├── dl_parallel.py             # 数据集多线程 Range 下载器
│   └── plot_results.py            # CPU 阶段曲线
├── experiments/toy/       # 2D/6D 机制验证
├── exp/                   # CPU 阶段实验输出
├── gpu_exp/               # 全预算 GPU 实验输出（CSV/flags/1M checkpoint）
├── results/               # 曲线 PNG、summary、gpu_summary、checkpoint 分析
├── logs/                  # 安装、下载、训练、GPU 运行日志
└── ppt/                   # 复现汇报 PPT 与生成脚本
```

## 快速开始

```powershell
# ---- Windows CPU 环境（已就绪）----
cd E:\FINO\FINO
$env:FINO_WANDB_MODE='offline'
E:\FINO\.venv\Scripts\python.exe main.py --env_name=pointmaze-medium-navigate-singletask-v0 `
  --offline_steps=8000 --online_steps=4000 --eval_interval=1000 --eval_episodes=10 `
  --log_interval=500 --save_interval=8000 --save_dir=E:/FINO/exp

# ---- WSL2 GPU 环境（全预算，2.3–2.5 h/任务）----
# 详细步骤见 wsl_setup/ 与 docs/02_reproduction_report.md §9，核心命令：
wsl -d FINO-Ubuntu -u root -- bash -lc "cd /opt/fino/FINO && \
  FINO_WANDB_MODE=offline XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 /opt/fino-venv/bin/python main.py \
  --env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \
  --offline_steps=1000000 --online_steps=500000 --eval_interval=50000 --eval_episodes=50 \
  --log_interval=5000 --save_interval=1000000 --save_dir=/mnt/e/FINO/gpu_exp"
```

## 重要说明

- **GPU 路径**：JAX 的 CUDA 插件只有 Linux 轮子；本机通过 WSL2 + Ubuntu 24.04 使用 RTX 5070（JAX 0.12.0 + CUDA）。代码仅需 1 处 distrax 可选导入适配。
- **CPU 路径**：Windows 原生 JAX 只能 CPU（4 处平台补丁）；3% 预算运行保留在 `exp/` 与报告 §7.2b 作为历史参照。
- **未覆盖**：D4RL 10 个任务（需 mujoco-py + MuJoCo 2.1）、像素环境与视频渲染。
- **复现预算参考**：单任务全预算 2.3–2.5 h；OGBench 默认 7 任务约 17–20 h；全量 55 任务约 5–9 天（单卡）。
