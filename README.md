# FINO 复现项目（Windows + CPU）

> 对 **ICLR 2026 论文** *Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning* 官方实现 **FINO** 的独立复现项目。
>
> - 上游仓库（论文作者）：<https://github.com/CTID282/FINO>
> - 论文：<https://openreview.net/forum?id=6wd38R8L0Z>
> - 本仓库性质：**非官方复现**，包含官方代码、4 处本地适配补丁、逐行技术分析、缩减预算实验、2D 机制验证、完整日志与复现报告。

---

## 1. 一句话结论

官方代码已在 **Windows 11 + CPU** 上完整跑通（训练 → 在线微调 → 评估 → checkpoint 保存/恢复），并完成了**论文任务的小预算部分复现**与**核心机制的定量验证**。

但 JAX 的 CUDA 插件只有 Linux 轮子，本机无 WSL/Docker/管理员权限，**RTX 5070 无法被 JAX 使用**；CPU 实测 64–102 ms/update，论文单任务 1.5M 步 ≈ 27 小时，50+ 任务 ≈ 55 天，因此**论文数值级全量复现不可行**。本仓库提供：[完整复现报告](docs/02_reproduction_report.md)（含 Linux + GPU 全量复现指南，约 200–450 GPU 小时）。

## 2. 仓库内容

| 路径 | 内容 |
|---|---|
| [`FINO/`](FINO/) | 官方源码 + 4 处 Windows/CPU 最小补丁（算法逻辑未改） |
| [`docs/01_code_analysis.md`](docs/01_code_analysis.md) | 724 行源码技术分析：方法→公式→代码行号映射、5 个 baseline 对比、16 条复现坑、README 超参表 |
| [`docs/02_reproduction_report.md`](docs/02_reproduction_report.md) | **复现总报告**：环境、补丁、实验设计、结果、偏差清单、全量复现指南 |
| [`docs/patches.diff`](docs/patches.diff) | 4 处补丁的完整 diff |
| [`docs/requirements-lock.txt`](docs/requirements-lock.txt) | 本机实际安装的 pip 包版本 |
| [`constraints.txt`](constraints.txt) | 钉版依赖（numpy 1.26.4 / jax 0.4.35 / flax 0.8.5 / ogbench 1.1.0 …） |
| [`experiments/toy/`](experiments/toy/) | 2D/6D 玩具实验：噪声注入、一步蒸馏、Q-guided 采样与熵自适应 β 的定量验证（含 8 张图、指标、[RESULTS.md](experiments/toy/RESULTS.md)） |
| [`scripts/`](scripts/) | 复跑脚本：`run_repro.ps1`、`dl_parallel.py`（限速服务器多线程下载）、`plot_results.py`、`analyze_checkpoint.py`、`bench.py`、`dataset_stats.py` |
| [`exp/`](exp/) | 三次运行的 `flags.json` / `train.csv` / `eval.csv`（不含二进制 checkpoint） |
| [`results/`](results/) | 训练/评估曲线、`summary.md`、checkpoint 级分析 |
| [`logs/`](logs/) | 安装、数据下载、训练、恢复测试日志（大日志已裁剪 tqdm 进度条，保留首尾各 300 行） |

## 3. 关键结果

### 3.1 论文任务 `antmaze-large-navigate-singletask-v0`（3% 预算缩减）

设置：30k 离线 + 10k 在线（论文为 1M + 500k），每 2.5k 步评估 10 episodes；FINO 与 FQL baseline 同预算对比。

| 指标 | FINO | FQL |
|---|---:|---:|
| BC flow loss（train，起点→终点） | 0.655 → 0.354 | 0.725 → 0.405 |
| 一步蒸馏 loss | 0.038 → 0.016 | 0.047 → 0.016 |
| 策略动作 Q 均值 | −6.1 → −85.6 | −6.1 → −85.7 |
| 最高评估成功率（170 episodes） | 0.1（25k 步，1/10） | 0.1（37.5k 步，1/10） |
| 耗时 / 速度 | 4,084 s / 102 ms/步* | 3,479 s / 87 ms/步* |

\* 两次运行并发抢 CPU。训练损失正常收敛、Q 值趋近理论下限 −100，说明实现正确；**3% 预算下两者都无法学会任务，单次成功属采样噪声，不作优劣结论**。

![antmaze-large FINO vs FQL](results/fig_eval_antmaze_compare.png)

### 3.2 端到端链路验证（pointmaze-medium，12k 步）

- 8k 离线 + 4k 在线，928 s；训练/验证损失、评估 CSV、wandb offline、Orbax checkpoint **保存 + 恢复续训**全部验证通过；
- 恢复测试：`--restore_path ... --restore_epoch=8000 --offline_steps=0 --online_steps=300` 成功。

### 3.3 2D 玩具机制验证（详见 [RESULTS.md](experiments/toy/RESULTS.md)）

| 机制 | 结论 |
|---|---|
| 一步蒸馏 | 学生 vs 10 步教师 EMD 0.0004；推理 **5.38× 加速**（1.44 ms vs 7.77 ms，N=4096） |
| Q-guided 采样 | d=6（K=3）时 per-state argmax Q +0.63、偏好模态占比 0.262→0.590；**d=2 时 K=1 机制退化为恒等** |
| 熵自适应 β | 实测 H≈−2.14 > 目标 −6，β 从 10 单调增到 21.3，方向与代码一致 |
| 噪声注入 | 官方指数 schedule η=0.1 在 t=0 仅 4.5e-6、t=1 才 0.1；玩具尺度上几乎无操作，方向（数据端平滑/正则）成立但量级偏弱 |

## 4. 环境搭建（Windows + CPU）

```powershell
# 1) 创建 venv（Python 3.11）
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# 2) 国内镜像 + 钉版依赖
$env:PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple'
$env:PIP_TRUSTED_HOST='pypi.tuna.tsinghua.edu.cn'
.venv\Scripts\python.exe -m pip install -c constraints.txt `
  "jax==0.4.35" "jaxlib==0.4.35" "flax==0.8.5" "optax==0.2.4" `
  "orbax-checkpoint==0.5.15" "distrax==0.1.5" "chex==0.1.87" "ml_collections==0.1.1" `
  "absl-py" "tqdm" "scikit-learn" "pillow" "matplotlib==3.8.4" "pandas" `
  "wandb" "imageio" "moviepy==1.0.3" "gymnasium==0.29.1" "mujoco==3.3.0" `
  "dm_control==1.0.20" "ogbench==1.1.0"
```

> 版本组合的踩坑记录见[报告 §4](docs/02_reproduction_report.md)：`ml_collections==0.1.7` 不存在、pip 会升级 numpy、`chex` 与 `optax` 冲突、flax 0.8.5 与 jax 0.4.38 的 tracing 不兼容（最终用 jax 0.4.35）。

## 5. 运行

```powershell
cd FINO
$env:FINO_WANDB_MODE='offline'   # 无 wandb 账号时使用本地离线日志

# 冒烟：pointmaze-medium（8k 离线 + 4k 在线，约 15 分钟）
..\.venv\Scripts\python.exe main.py `
  --env_name=pointmaze-medium-navigate-singletask-v0 --agent.alpha=10 `
  --offline_steps=8000 --online_steps=4000 --eval_interval=1000 --eval_episodes=10 `
  --log_interval=500 --save_interval=8000 --save_dir=../exp

# 论文任务缩减实验（约 70 分钟；CPU）
..\.venv\Scripts\python.exe main.py `
  --env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min `
  --offline_steps=30000 --online_steps=10000 --eval_interval=2500 --eval_episodes=10 `
  --log_interval=1000 --save_interval=30000 --save_dir=../exp

# 生成图表与汇总
..\.venv\Scripts\python.exe ..\scripts\plot_results.py
..\.venv\Scripts\python.exe ..\scripts\analyze_checkpoint.py
```

数据集：OGBench 首次运行自动下载到 `~/.ogbench/data`（官方服务器较慢，可用 [`scripts/dl_parallel.py`](scripts/dl_parallel.py) 多线程 Range 下载）。本仓库**不包含**数据集与二进制 checkpoint（`.venv`、`exp/**/30000` 等已被 `.gitignore` 排除）。

## 6. 上游代码改动（4 处，最小侵入）

完整 diff 见 [`docs/patches.diff`](docs/patches.diff)。均不改变算法/训练逻辑：

1. `main.py`：`MUJOCO_GL=egl` 仅在非 Windows 平台设置（原生 Windows 无 EGL）；
2. `utils/log_utils.py`：wandb 默认 `offline`（可用 `FINO_WANDB_MODE` 覆盖）；
3. `utils/log_utils.py`：删除新版 wandb 已移除的 `Settings(start_method=...)`；
4. `utils/evaluation.py`：`np.random.randint(0, 2**32)` → `2**31-1`（Windows 上 int32 上界溢出）。

## 7. 复现偏差与已知问题

- 预算缩减到论文的 3%，成功率为 0–10%，**不能与论文表格数值比较**；
- CPU vs GPU 的浮点差异、ogbench 1.1.0 版本固定；
- D4RL 的 10 个任务（`d4rl`/`mujoco-py`/`gym==0.23.1`）在 Windows 上不可安装，未覆盖；
- 源码审计发现（详见[报告 §8](docs/02_reproduction_report.md)）：噪声注入实际 schedule 与论文式(7)不一致、β 更新符号与论文式(9)方向相反、`to_eval` 在 batch>1 时全局 argmax、CsvLogger 固定表头丢列、β 不随 checkpoint 保存。

## 8. 引用与致谢

```bibtex
@inproceedings{shin2026flow,
  title={Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning},
  author={Yongjae Shin and Jongseong Chae and Jongeui Park and Youngchul Sung},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=6wd38R8L0Z}
}
```

- 官方代码：<https://github.com/CTID282/FINO>（本仓库 `FINO/` 为其副本 + 4 处适配补丁）
- 代码母体 FQL：<https://github.com/seohongpark/fql>
- 数据集：[OGBench](https://github.com/seohongpark/ogbench)（<https://rail.eecs.berkeley.edu/datasets/ogbench>）

> 上游仓库未附 LICENSE；本复现项目仅用于学习与研究，官方代码版权归原作者所有。
