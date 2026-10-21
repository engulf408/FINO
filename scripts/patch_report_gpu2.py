# -*- coding: utf-8 -*-
"""Second report patch: §9 (GPU reproduction path) and §10 (deliverables)."""
import io

P = r'E:\FINO\docs\02_reproduction_report.md'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:80]
    s = s.replace(old, new, 1)


old9 = s[s.index('## 9. 如何在 Linux + GPU 上全量复现（推荐路径）'):s.index('---\n\n## 10. 交付物清单')]
new9 = '''## 9. 在 Linux + GPU 上复现：本机已走通的实际路径

> 本节路径已于 2026-10-21 在 Windows 11 + WSL2 + RTX 5070 上实际执行成功；脚本在 `wsl_setup/`，日志在 `logs/`。

1. **WSL2 + Ubuntu**：管理员 PowerShell 启用 `Microsoft-Windows-Subsystem-Linux` / `VirtualMachinePlatform` → 重启 →
   `winget install Microsoft.WSL` → 下载 Ubuntu 24.04 base rootfs → `wsl --import FINO-Ubuntu E:\\FINO\\wsl\\FINO-Ubuntu <rootfs> --version 2`。
2. **系统依赖**：apt 镜像切清华（先 HTTP 装 `ca-certificates`，再 HTTPS）；安装 `python3(3.12) python3-venv python3-pip git curl libegl1 libgl1 libosmesa6`。
3. **Python 环境**：`python3 -m venv /opt/fino-venv`；pip 切清华镜像；
   `pip install "jax[cuda12]"`（实测 0.12.0，支持 Blackwell/sm_120）、`flax optax orbax-checkpoint distrax chex ml_collections absl-py tqdm scikit-learn pillow matplotlib pandas wandb imageio moviepy "gymnasium==0.29.1" "mujoco==3.3.0" "dm_control==1.0.20" ogbench`。
4. **代码**：`cp -r /mnt/e/FINO/FINO /opt/fino/FINO`；唯一适配：`utils/networks.py` 的 distrax 可选导入（补丁 #5）。
5. **数据**：OGBench 首次运行自动下载到 `~/.ogbench/data`（也可用 `scripts/dl_parallel.py` 多线程预下载后拷贝）；
   D4RL 10 个任务需额外安装 `d4rl`/`mujoco-py` 与 MuJoCo 2.1 二进制（本机未覆盖）。
6. **运行（论文原设置）**：

   ```bash
   cd /opt/fino/FINO
   export FINO_WANDB_MODE=offline XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
   # FINO
   python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \\
       --offline_steps=1000000 --online_steps=500000 --eval_interval=50000 --eval_episodes=50 \\
       --log_interval=5000 --save_interval=1000000 --save_dir=/mnt/e/FINO/gpu_exp
   # FQL baseline（同预算）
   python main.py --agent=agents/fql.py --env_name=antmaze-large-navigate-singletask-v0 \\
       --agent.alpha=10 --agent.q_agg=min --offline_steps=1000000 --online_steps=500000 \\
       --eval_interval=50000 --eval_episodes=50 --log_interval=5000 --save_interval=1000000 --save_dir=/mnt/e/FINO/gpu_exp
   ```

7. **已实测成本**（RTX 5070 12 GB）：单任务全预算 **2.3–2.5 小时**（含 31 次 × 50 episodes 评估）；
   纯训练 update 1.58 ms（FINO）/1.65 ms（FQL）。OGBench 默认 7 任务约 **17–20 小时**，55 任务约 **5–9 天**（单卡）。
8. **验收**：`eval.csv` 最后一行 `evaluation/success` 与论文表格对比；消融可加 `--agent.noise_scale=0`（关噪声注入）、
   调整 `flow_steps`、或改 `--agent.q_agg=mean`。

'''
s = s.replace(old9, new9, 1)

# ---- §10 ----
rep("| `E:\\FINO\\FINO` | 官方源码 + 4 处 Windows/CPU 补丁（算法未改） |",
    "| `E:\\FINO\\FINO` | 官方源码 + 5 处平台适配补丁（4 处 Windows/CPU + 1 处 GPU，算法未改） |")
rep("| `E:\\FINO\\_extract\\FINO-main` | 未改动的官方源码备份（用于 diff/对照） |",
    "| `E:\\FINO\\FINO`（WSL 副本 `/opt/fino/FINO`） | 运行副本（distrax 可选导入补丁） |\n"
    "| `E:\\FINO\\_extract\\FINO-main` | 未改动的官方源码备份（用于 diff/对照） |")
rep("| `docs/patches.diff` | 全部补丁 diff |",
    "| `docs/patches.diff` | 全部补丁 diff（含 GPU 栈第 5 处，见 `wsl_setup/`） |")
rep("| `scripts/dataset_stats.py` | 数据集统计 |",
    "| `scripts/dataset_stats.py` | 数据集统计 |\n"
    "| `scripts/bench_gpu.py` | GPU 速度基准（FINO/FQL/环境） |\n"
    "| `scripts/analyze_gpu_results.py` | 全预算 GPU 结果汇总与曲线 |\n"
    "| `scripts/analyze_ckpt_full.py` | 全预算 checkpoint 级分析 |")
rep("| `exp/FINO/...` | 各运行的 `train.csv/eval.csv/flags.json/checkpoint` |",
    "| `exp/FINO/...` | CPU 阶段各运行的 `train.csv/eval.csv/flags.json/checkpoint` |\n"
    "| `gpu_exp/FINO/...`、`gpu_exp/FQL/...` | **全预算 GPU 运行**：train.csv/eval.csv/flags.json/checkpoint(1M) |\n"
    "| `wsl_setup/` | WSL2 安装与运行脚本（install/finish/setup1-5/launch_gpu_runs/launch_batch2） |")
rep("| `results/` | 训练/评估曲线 PNG、`summary.csv`、`summary.md`、`dataset_stats.json` |",
    "| `results/` | CPU/GPU 训练评估曲线、`summary.csv`、`summary.md`、`gpu_summary.csv`、`checkpoint_analysis*.json` |")
rep("| `logs/` | 安装、下载、训练、恢复测试的完整日志 |",
    "| `logs/` | 安装、下载、训练、恢复测试、GPU 运行与下载的完整日志 |")

io.open(P, 'w', encoding='utf-8').write(s)
print('report patched (9/10) ok, lines:', len(s.splitlines()))
