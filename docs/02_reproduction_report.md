# FINO（ICLR 2026）复现报告

> 仓库：<https://github.com/CTID282/FINO>
> 论文：*Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning*（ICLR 2026，OpenReview `6wd38R8L0Z`）
> 本机复现目录：`E:\FINO`（源码在 `E:\FINO\FINO`，环境在 `E:\FINO\.venv`）
> 报告日期：2026-10-20
> 配套材料：`docs/01_code_analysis.md`（源码逐行分析）、`docs/patches.diff`（为跑通而做的 4 处最小改动）、`experiments/toy/RESULTS.md`（2D 机制验证）、`results/`（曲线与汇总表）、`logs/`（全部运行日志）

---

## 0. 结论摘要（TL;DR）

1. **官方代码已在 Windows 11 + CPU 上完整跑通**：环境/数据集加载 → FINO 三网络初始化 → 离线训练 → 在线微调（含 Q-guided 采样与熵自适应 β）→ 周期评估 → Orbax checkpoint 保存/恢复，全链路都有实测日志与曲线。
2. **无法在本机做"论文数值级"复现**：JAX 的 CUDA 插件只有 Linux 轮子，本机 Windows 无 WSL/Docker/管理员权限，**RTX 5070 无法被 JAX 使用**，只能 CPU。实测 antmaze-large 约 **15.5 update/s（64 ms/step）**，论文单任务 1.5M 步 ≈ **26.7 小时**，50+ 任务 ≈ **55 天**（单进程），不可行。
3. **交付的是"代码/方法级复现 + 小预算部分复现"**：
   - 端到端冒烟实验（pointmaze-medium，8k 离线 + 4k 在线，12k 步，15.5 分钟）验证了训练/评估/日志/checkpoint 全链路；
   - 论文任务 antmaze-large-navigate-singletask-v0 做了 30k 离线 + 10k 在线的缩减预算实验（约 3% 论文预算），曲线与失败/成功情况如实报告；
   - 2D 玩具实验定量验证了 FINO 的三个核心机制（带噪 flow matching、一步蒸馏、Q-guided 采样 + 熵自适应 β），并发现两个值得作者注意的细节（见 §7.3）。
4. **发现并修正了 4 个"非 Linux/非旧版依赖"环境下会直接崩溃的问题**（`MUJOCO_GL=egl`、wandb 在线模式、新版 `wandb.Settings.start_method` 已被删除、`np.random.randint(0, 2**32)` 在 Windows 上 int32 溢出）。改动见 `docs/patches.diff`，均为最小侵入，不改算法逻辑。
5. 附带对源码的静态审计发现若干与论文表述不一致或潜在的实现问题（噪声注入时间表、β 更新符号、`to_eval` 的 batch>1 argmax、CsvLogger 丢列等），见 §8，供作者与后续复现者参考。

---

## 1. 复现目标与范围

论文的核心贡献是在 FQL（flow Q-learning）基础上做两处修改：

| 修改 | 论文位置 | 代码位置 | 一句话说明 |
|---|---|---|---|
| **Noise-Injected Flow Matching** | §4.1，Eq.(7) | `agents/fino.py:56-58` | 在插值点 `x_t` 上按时间表 `α_t` 注入高斯噪声后再训练向量场，使 `t=1`（数据端）的边际方差变为 `η²`，扩大一步策略的支持集，为在线探索提供"数据之外一点"的动作。 |
| **Q-guided sampling + 熵自适应 β** | §4.2，Eq.(8)(9) | `agents/fino.py:153-197, 215-232, 307-335`；`main.py:220-224` | 在线阶段对每个状态采多个噪声候选动作，用 `softmax(β·Q)` 采样（评估时 argmax）；β 由 GMM 估计的策略熵反馈调节，自适应探索/利用。 |

工作分四块：

1. 环境与代码：在尽量少改动的前提下让官方实现跑起来（§4、§5）。
2. 端到端验证：跑通论文的 offline→online 流程与全部工程组件（§6.1、§7.1）。
3. 论文任务的部分复现：在 antmaze-large 上做预算缩减实验（§6.2、§7.2）。
4. 机制验证与复现成本核算：2D 玩具实验 + CPU 计时外推，给出在 Linux+GPU 上全量复现的可行方案（§7.3、§7.4、§9）。

**不在范围内**：D4RL 系列（`d4rl`/`mujoco-py`/`gym==0.23.1` 在 Windows 上基本不可安装）、像素环境（`impala_small` 编码器/视频渲染）、全部 50+ 任务的完整训练。

---

## 2. 本机环境与硬件

| 项目 | 值 |
|---|---|
| OS | Microsoft Windows 11 家庭版 25H2（build 26100） |
| CPU | AMD Ryzen 7 9700X（8 核 / 16 线程） |
| RAM | 31.1 GB |
| GPU | NVIDIA GeForce RTX 5070 12 GB（驱动 617.14，CUDA UMD 13.4）——**JAX 用不到** |
| 系统级虚拟化 | WSL 未安装；无 Docker、无管理员权限（`IsInRole(Administrator)=False`） |
| Python | 3.11.16（venv 由本机已有 `E:\anaconda\envs\dl\python.exe` 创建） |

### 2.1 为什么 GPU 用不上（关键限制）

- JAX 官方仅对 **Linux** 发布 CUDA 插件。已核实 PyPI：`jax-cuda12-plugin`、`jax-cuda13-plugin`、`jax-cuda12-pjrt`、`jax-cuda13-pjrt` 的最新版本（0.12.0）**只有 manylinux 轮子，没有 Windows 轮子**；`jaxlib` 本身有 `win_amd64` 轮子，但只是 CPU 后端。
- 本机没有 WSL2（`wsl.exe --status` 提示未安装），也没有 Docker；安装 WSL 需要管理员权限和重启，当前会话无法完成。
- 因此 JAX 只能走 CPU：`jax.devices()` 返回 `[CpuDevice(id=0)]`。

### 2.2 CPU 上的实测速度（决定复现规模）

| 场景 | 实测 |
|---|---|
| FINO 单次 update（29 维观测、batch=256、512×512×4 网络、10 步 flow） | **64 ms**（antmaze 训练循环内实测 15.5 it/s） |
| FINO 单次 update（2 维观测，pointmaze） | 77 ms（12.9 it/s） |
| FQL 单次 update（同上） | 约 25–40 ms |
| pointmaze 环境 step | 0.05–0.08 ms |
| antmaze 环境 step | 0.26 ms |

外推：论文每个任务 = 1M 离线 + 500k 在线 = **1.5M 步** → 64 ms × 1.5M ≈ **26.7 小时/任务**；OGBench 默认 7 任务 ≈ 7.8 天，README 全量 45+10 任务 ≈ **55 天**（单进程）。即使用 8 个进程并行（本机 8 核），也需约 7 天，且每个进程只有 1 核速度会更慢。**故完整复现不在本机可行范围内**。

---

## 3. 仓库与方法速览

详细逐行分析见 `docs/01_code_analysis.md`（724 行），这里只摘复现必需的部分。

### 3.1 文件结构

```
FINO/
├── main.py                # 唯一入口：flags + ml_collections 配置、环境/数据集、offline→online 主循环、评估、保存
├── agents/
│   ├── fino.py            # FINO 全部算法（critic/actor 损失、Q-guided 采样、熵自适应 β）
│   ├── fql.py             # FQL baseline（FINO 的母体）
│   ├── ifql.py / iql.py / sac.py / rebrac.py
│   └── __init__.py        # 注册表
├── utils/
│   ├── networks.py        # MLP / Value(集成Q) / Actor / ActorVectorField（flow 向量场）
│   ├── flax_utils.py      # ModuleDict / TrainState / Orbax save+restore
│   ├── datasets.py        # Dataset / ReplayBuffer
│   ├── evaluation.py      # evaluate()
│   └── log_utils.py       # CsvLogger / wandb
├── envs/env_utils.py      # OGBench / D4RL 分派
└── requirements.txt
```

### 3.2 FINO 的训练目标（单次 update，一次 Adam 更新）

```
critic_loss = MSE( Q(s,a), r + γ·mask·Q_target(s', a'~π_one-step(s')) )
actor_loss  = bc_flow_loss + α · distill_loss + q_loss
其中：
  bc_flow_loss = || v_θ(x_t, t | s) − (x_1 − x_0) ||² ,  x_t = (1−t)x_0 + t x_1,  x_1 = a_data
  注入噪声：x_t' = x_t + η·exp(10(t−1))·ε ,  ε~N(0,I)   （fino.py:56-58, η=0.1）
  distill_loss = || π_one-step(s, ε) − SG[ 10-step Euler 采样 π_flow(s, ε) ] ||²
  q_loss       = −mean_i Q_i(s, clip(π_one-step(s, ε)))
```

三个关键结论（复现时容易搞错）：

1. **噪声注入在数据端（t→1），不是起点（t→0）**。代码 `noise_scale = 0.1·exp(10(t−1))`：t=0 时 ≈4.5e-6，t=1 时为 0.1。这与论文 §4.1 的边界条件 `α_0²=0, α_1²=η²` 一致；`t=0` 的"带噪起点"指的是基底 `x_0~N(0,I)`。
2. actor 的 Q 项**对 critic 停止梯度**，且用集成均值；`q_agg=min` 只影响 TD 目标与候选动作选择。
3. 在线采集用 `temperature=1`（softmax(βQ) 采样），评估用 `temperature=0`（argmax），**β 只影响采集，不影响评估动作**；β 不被 checkpoint 保存（恢复后重置为 10）。

---

## 4. 环境搭建（可复现命令）

官方 `requirements.txt` 针对 Python 3.9 + Linux + CUDA。本机实际使用的钉版组合如下（`E:\FINO\constraints.txt`）：

```
numpy==1.26.4      scipy==1.13.1     jax==0.4.35        jaxlib==0.4.35
ml_dtypes==0.4.0   flax==0.8.5       optax==0.2.4       orbax-checkpoint==0.5.15
distrax==0.1.5     chex==0.1.87      tensorflow_probability==0.24.0
gymnasium==0.29.1  mujoco==3.3.0     dm_control==1.0.20 matplotlib==3.8.4
ogbench==1.1.0     ml_collections==0.1.1
```

搭建步骤（本机实际执行）：

```powershell
# 1) 用已有 Python 3.11 建 venv（本机无网络问题以外的系统依赖）
E:\anaconda\envs\dl\python.exe -m venv E:\FINO\.venv
E:\FINO\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# 2) 用国内镜像 + 约束文件安装（直连 PyPI 实测仅 28 KB/s，装不完）
$env:PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple'
$env:PIP_TRUSTED_HOST='pypi.tuna.tsinghua.edu.cn'
E:\FINO\.venv\Scripts\python.exe -m pip install -c E:\FINO\constraints.txt `
  "jax==0.4.35" "jaxlib==0.4.35" "flax==0.8.5" "optax==0.2.4" `
  "orbax-checkpoint==0.5.15" "distrax==0.1.5" "chex==0.1.87" "ml_collections==0.1.1" `
  "absl-py" "tqdm" "scikit-learn" "pillow" "matplotlib==3.8.4" "pandas" `
  "wandb" "imageio" "moviepy==1.0.3" "gymnasium==0.29.1" "mujoco==3.3.0" `
  "dm_control==1.0.20" "ogbench==1.1.0"
```

踩过的坑与解决办法：

| 现象 | 原因 | 解决 |
|---|---|---|
| `ml_collections==0.1.7` 不存在 | 该版本未发布到 PyPI | 用 `0.1.1` |
| 安装后 numpy 被升到 2.4.6 | pip 新版解析器升级传递依赖 | 用 `constraints.txt` 钉死 numpy/jax 等 |
| `chex==0.1.86` 与 `optax==0.2.4` 冲突 | optax 0.2.4 要求 `chex>=0.1.87` | 改钉 `chex==0.1.87` |
| flax 0.8.5 + jax 0.4.38 报 `'EvalTrace' object has no attribute 'level'` | jax 0.4.38 改了 tracing 内部 API | 降到 **jax/jaxlib 0.4.35** |
| `ogbench` 无 `__version__` | 包未实现版本属性 | 无需处理（不影响功能） |

---

## 5. Windows / CPU 适配补丁（4 处，最小侵入）

完整 diff 见 `docs/patches.diff`。原版备份在 `E:\FINO\_extract\FINO-main`（未改动的官方源码）。

| # | 文件 | 改动 | 原因 |
|---|---|---|---|
| 1 | `main.py:22` | `os.environ["MUJOCO_GL"]="egl"` 改为仅非 Windows 设置 | 原生 Windows 没有 EGL；该变量只在渲染时生效，本复现无视频，但保留正确语义 |
| 2 | `utils/log_utils.py:68` | `mode='online'` → `mode=os.environ.get('FINO_WANDB_MODE','offline')` | 无 wandb 账号/网络；离线模式只写本地 `wandb/`，`wandb.log` 行为不变 |
| 3 | `utils/log_utils.py:84` | 删去 `wandb.Settings(start_method='thread')` | wandb 新版已移除 `start_method`，否则 pydantic 校验直接报错 |
| 4 | `utils/evaluation.py:60` | `np.random.randint(0, 2**32)` → `np.random.randint(0, 2**31-1)` | Windows 上 numpy 默认整型为 int32，上界 2^32 越界（`ValueError: high is out of bounds for int32`） |

四处都不改变算法/训练逻辑；除第 3 处外，在 Linux 上也不会破坏原行为（第 2 处可用环境变量 `FINO_WANDB_MODE=online` 恢复）。

---

## 6. 复现实验设计

### 6.1 数据集

用 OGBench 1.1.0 自带下载器下载（官方服务器 `rail.eecs.berkeley.edu` 单连接仅 ~18 KB/s，本机用 24–40 路 HTTP Range 并发下载并校验 `Content-Length` + `np.load().files`，脚本见 `scripts/dl_parallel.py`）：

| 数据集 | 文件大小 | transitions | obs 维 | action 维 | 轨迹长度 | episodes |
|---|---:|---:|---:|---:|---:|---:|
| `pointmaze-medium-navigate-v0`（冒烟测试） | 20.2 MB（val 2.0 MB） | 1,001,000 | 2 | 2 | 1001 | 1000 |
| `antmaze-large-navigate-v0`（**论文任务**） | 242.9 MB（val 24.3 MB） | 1,001,000 | 29 | 8 | 1001 | 1000 |

`*-singletask-v0` 数据集名会被 ogbench 自动映射到上述文件名，并按固定目标 relabel `rewards/masks`（`ogbench/relabel_utils.py`）：导航任务 `reward = success − 1`、`mask = 1 − success`。

### 6.2 运行预算

| 运行 | 离线步数 | 在线步数 | 评估 | 说明 |
|---|---:|---:|---|---|
| 冒烟 `pointmaze-medium`（FINO） | 8,000 | 4,000 | 每 1k 步 × 10 episodes | 链路验证（含 d=2 时 Q-guided 采样退化的边界情况） |
| 论文任务 `antmaze-large`（FINO） | 30,000 | 10,000 | 每 2.5k 步 × 10 episodes | 论文预算的 3%，超参沿用 README（`alpha=10, q_agg=min`） |
| checkpoint 恢复测试 | 0 | 300 | 每 100 步 × 2 episodes | 从 pointmaze 第 8000 步 checkpoint 恢复后继续在线训练 |
| （可选）baseline `antmaze-large`（FQL） | 30,000 | 10,000 | 同 FINO | 验证 `--agent=agents/fql.py` 切换与同预算对比 |

评估 episodes 与间隔都做了缩减（论文为每 50k 步 × 50 episodes），这是纯工程性缩减，不改变算法。

---

## 7. 结果

### 7.1 端到端链路验证（pointmaze-medium，12k 步）

运行命令（`--save_dir` 绝对路径，避免依赖 cwd）：

```powershell
cd E:\FINO\FINO
$env:FINO_WANDB_MODE='offline'
E:\FINO\.venv\Scripts\python.exe main.py `
  --env_name=pointmaze-medium-navigate-singletask-v0 --agent.alpha=10 `
  --offline_steps=8000 --online_steps=4000 `
  --eval_interval=1000 --eval_episodes=10 --log_interval=500 `
  --save_interval=8000 --save_dir=E:/FINO/exp
```

`docs/01_code_analysis.md` 中列出的所有工程组件都在这次运行中被实际触发并验证：

- 训练 12,000 步（离线 8,000 + 在线 4,000），耗时 928 s（77 ms/步），wandb offline 正常写入；
- `train.csv` 每 500 步记录训练 + 验证损失、梯度统计、耗时；
- `eval.csv` 每 1,000 步记录 10 episodes 的 `success/return/...`；
- 第 8,000 步 Orbax checkpoint 保存成功（4 个模块目录）；
- 用 `--restore_path/--restore_epoch=8000 --offline_steps=0 --online_steps=300` 从 checkpoint 恢复并继续在线训练 300 步成功（exit=0），验证了 `save_agent/restore_agent`；
- 在线阶段（i>8000）每 1,000 步调用 `update_entropy`（GMM 熵估计 + β 更新），无异常（注意：CsvLogger 表头在首行固定，后加的 `evaluation/beta|entropy` 列不会写入 CSV，见 §8）。

训练指标（`train.csv`，step 1 → 12,000）：

| 指标 | 起点 | 终点 |
|---|---:|---:|
| BC flow loss（train） | 1.0868 | 1.0221 |
| BC flow loss（val） | — | 0.9078 |
| 一步蒸馏 loss | 0.0264 | 0.0131 |
| critic TD loss | 0.0792 | 0.2835 |
| 策略动作 Q 均值 | −3.20 | −45.91 |

**评估结果：12 次评估（step 1…12,000）success 全部为 0，episode return 恒为 −1000。** 这在缩减预算下是预期内的：pointmaze-medium 的示范数据需要大量更新才能学到有效导航策略（论文每任务为 1M 离线步，本实验仅 8k，约 2 个 epoch）；该任务用于验证"流程正确"，而非追求性能。曲线见 `results/fig_training_pointmaze.png`、`results/fig_eval_pointmaze.png`：

![pointmaze 训练曲线](../results/fig_training_pointmaze.png)

### 7.2 论文任务 antmaze-large-navigate-singletask-v0（缩减预算）

> **实验设置。** 论文原设置：1M 离线 + 500k 在线，每 50k 步评估 50 episodes。本实验缩减为：**30,000 离线 + 10,000 在线，每 2,500 步评估 10 episodes**（约论文预算的 3%），超参沿用 README（`--agent.alpha=10 --agent.q_agg=min`）。为做同预算对比，同一设置下又跑了官方仓库自带的 **FQL baseline**（`--agent=agents/fql.py`，与 FINO 的唯一区别是不含噪声注入与 Q-guided/β 机制）。

运行命令：

```powershell
cd E:\FINO\FINO; $env:FINO_WANDB_MODE='offline'
# FINO
E:\FINO\.venv\Scripts\python.exe main.py --env_name=antmaze-large-navigate-singletask-v0 `
  --agent.alpha=10 --agent.q_agg=min --offline_steps=30000 --online_steps=10000 `
  --eval_interval=2500 --eval_episodes=10 --log_interval=1000 --save_interval=30000 --save_dir=E:/FINO/exp
# FQL baseline
E:\FINO\.venv\Scripts\python.exe main.py --agent=agents/fql.py --env_name=antmaze-large-navigate-singletask-v0 `
  --agent.alpha=10 --agent.q_agg=min --offline_steps=30000 --online_steps=10000 `
  --eval_interval=2500 --eval_episodes=10 --log_interval=1000 --save_interval=30000 --save_dir=E:/FINO/exp
```

**训练过程指标**（40,000 步，每个数字来自 `train.csv`）：

| 指标（起点 → 终点） | FINO | FQL |
|---|---:|---:|
| BC flow loss（train） | 0.655 → 0.354 | 0.725 → 0.405 |
| BC flow loss（val，终点） | 0.383 | 0.404 |
| 一步蒸馏 loss | 0.0385 → 0.0158 | 0.0468 → 0.0157 |
| critic TD loss（终点） | 0.191 | 0.253 |
| 策略动作 Q 均值 | −6.11 → −85.62 | −6.12 → −85.66 |
| 总耗时 / 速度 | 4,084 s / 102.1 ms/update\* | 3,479 s / 87.0 ms/update\* |

\* 两次运行并发抢 CPU，单跑实测更快（FINO 约 64 ms/update，见 §2.2）。训练损失均正常收敛（BC flow 与蒸馏 loss 都持续下降），说明**训练链路本身健康**。

**评估结果**（每 2,500 步 × 10 episodes，共 17 次评估 = 170 episodes）：

| 评估点 | FINO success | FINO return | FQL success | FQL return |
|---|---:|---:|---:|---:|
| 1 … 22,500 | 0.0 | −1000.0 | 0.0 | −1000.0 |
| 25,000 | **0.1** | **−968.9** | 0.0 | −1000.0 |
| 27,500 … 35,000 | 0.0 | −1000.0 | 0.0 | −1000.0 |
| 37,500 | 0.0 | −1000.0 | **0.1** | **−985.5** |
| 40,000 | 0.0 | −1000.0 | 0.0 | −1000.0 |

**结论（诚实版）**：

1. 在 3% 预算下，**FINO 与 FQL 都基本学不会 antmaze-large**（170 个评估 episode 中各自只有 1 个成功，且集中在个别评估点），与论文中 1M 步离线 + 500k 步在线的数量级差距一致——这个结果**不能**用来比较两者优劣，只能证明两条代码路径都能完整训练到 40k 步并正常评估。
2. 成功事件出现在 25k（FINO）与 37.5k（FQL）各 1 次，属于 10 episode 采样下的随机波动（10% 的分辨率、单 seed），不构成"FINO 更好"的证据。
3. 训练曲线显示 BC flow/蒸馏 loss 稳步下降、Q 值向理论值（−1/(1−γ)≈−100）靠近，说明**实现层面无异常**；性能瓶颈是预算，不是代码。
4. 评估/训练曲线见 `results/fig_eval_antmaze_compare.png`、`results/fig_training_antmaze_fino_fino.png`、`results/fig_training_antmaze_fql_fql.png`。

**从中期 checkpoint（第 30,000 步）做的策略质量分析**（`results/checkpoint_analysis.json`，256 个验证状态）：

| 指标 | FINO | FQL | 说明 |
|---|---:|---:|---|
| 一步策略 vs 10 步教师的能量距离（ED） | 0.0078 | 0.0076 | 一步蒸馏很好地逼近多步 flow，与玩具实验 B 一致 |
| 一步策略 vs 数据集动作的 MSE | 0.1182 | 0.1165 | 与数据动作仍有明显差距（欠拟合） |
| 10 步教师 vs 数据集动作的 MSE | 0.0767 | 0.0764 | 教师也远未拟合（3% 预算下预期内） |
| 贪心策略动作的平均 Q 值 | −78.85 | −78.74 | 两者接近理论下限 −100，说明 critic 已能区分好坏动作 |

该分析再次确认：**实现正确、蒸馏有效，但预算不足导致模仿/RL 都远未收敛**。

![antmaze-large FINO vs FQL 评估对比](../results/fig_eval_antmaze_compare.png)

![antmaze-large FINO 训练曲线](../results/fig_training_antmaze_fino_fino.png)


### 7.3 2D 玩具实验：三个核心机制的定量验证

为规避 GPU 限制、直接检验论文主张，用仓库中的 `utils/networks.py`（`ActorVectorField`）与 `agents/fino.py` 的损失公式，在合成 2D/6D 数据上做了三组实验（详情 `experiments/toy/RESULTS.md`，脚本与图在 `experiments/toy/`）：

**A. 噪声注入（BC flow matching）**

- 官方 exp schedule（η=0.1）：与"无注入"相比，动作分布 EMD 0.02929 → 0.02943（+0.5%），t≤0.1 处向量场 MSE 比值 1.003，Lipschitz(t=0.9) 1.020 → 1.008（−1.1%）；
- 说明：官方时间表在 t=0 注入量仅 4.5e-6，本质是**数据端**正则；在该玩具尺度上 η=0.1 的效果接近无操作；
- 若改成论文 Eq.(7) 的二次型方差 schedule（`α_t²=(η²−2η)t²+2ηt`），Lipschitz(t=0.9) 降 29%，但 EMD 变差 +6.7%、t≥0.9 的向量场 MSE +13.4%——方向对但会损失保真；
- η 扫描：η=0.1/0.3/1.0 → Lipschitz(t=0.9)=1.008/0.943/0.545，EMD=0.0294/0.0300/0.0393。**结论：注入强度与"平滑/支持集扩大"正相关，与"分布保真"负相关，官方 η=0.1 偏保守。**

**B. 一步蒸馏**

- 学生 vs 教师 EMD = 0.00039（MMD 6.4e-5）；学生 vs 数据 0.0323（教师 0.0315，仅 +2.7%）；
- 推理耗时 N=4096：一步 1.44 ms vs 10 步教师 7.77 ms（**5.38× 加速**）；
- flow_steps 1/2/5/10 → 采样 EMD 0.321/0.170/0.0478/0.0315；
- α 扫描：α=1 时 Q 提升但分布坍缩为单模态；α=100/1000 时保真且 Q 仅小幅优化 → **支持 README 中按任务上调 α（3~10000）的设计**。

**C. Q-guided 采样 + 熵自适应 β**

- `num_samples = min(10, (action_dim+1)//2)`：**action_dim=2 时 K=1，选择机制严格退化为恒等**（identity=1.000，|ΔQ|=0）——这解释了为什么冒烟实验中 Q-guided 采样"看起来没生效"；论文任务 action_dim=8 时 K=4 才会生效；
- action_dim=6（K=3）：per-state argmax Q 4.851 → 5.483（+0.632），偏好模态占比 0.262 → 0.590；env β=10 的 softmax 采样 +0.503 且保持更多多样性；
- 熵自适应 β：实测策略熵 H≈−2.14 > 目标 −6，β 经 30 次更新 10 → 21.3，H 缓降，方向与代码 `β += 0.1(H+d)` 一致；
- **额外发现**：`to_eval` 分支用 `actions[jnp.argmax(q)]`（无 axis），当 batch>1 时是**全局 argmax**（所有状态共享同一噪声索引），语义错误；官方评估逐条 B=1 调用因此不受影响，但 batch 评估会静默退化。见 §8。

### 7.4 计时与全量复现成本

| 项目 | 实测/外推 |
|---|---|
| FINO update（antmaze 规模） | 64 ms（15.5 it/s） |
| 论文单任务（1.5M 步） | ≈ 26.7 h（本机 CPU） |
| OGBench 默认 7 任务 | ≈ 7.8 天 |
| README 全量 45（OGBench）+10（D4RL）任务 | ≈ 55 天 |
| 8 进程并行（8 核） | 每进程约 1 核，合计吞吐提升有限，仍约 1 周以上 |

---

## 8. 与论文的差异、实现审计与已知问题

### 8.1 复现偏差（诚实清单）

1. **预算**：离线/在线步数缩减到论文的 0.4%~3%，评估 episodes 与间隔缩减；这直接导致成功率为 0，**不能**与论文表格数值比较。
2. **硬件**：CPU 而非 GPU；浮点运算顺序/精度差异（XLA CPU vs CUDA）会造成微小差异，但对 512×512 MLP 影响可忽略。
3. **超参**：沿用 README 的 `alpha`、`q_agg`、`discount`，未重新调参；`beta` 初值/更新率按默认。
4. **环境/数据版本**：ogbench 1.1.0（README 钉的版本），但本机 Python 3.11/jax 0.4.35 与原作者的 Python 3.9/jax 版本不同；OGBench 文档提到 1.2.0 起 singletask 的 reward/terminated 计算时点由 `s'` 改为 `s`，1.1.0 不受影响，但与"最新版本"复现者会有差异。
5. **D4RL 未覆盖**：antmaze-umaze 等 10 个 D4RL 任务需要 `d4rl + mujoco-py + gym==0.23.1`，Windows 无法构建。
6. **wandb 离线**：无云端曲线，所有指标以 CSV 形式保存（`exp/FINO/<env>/0/*.csv`）。
7. **视频渲染未验证**：`video_episodes=0`；Windows 需设 `MUJOCO_GL=wgl/glfw`，且 `utils/log_utils.py:get_wandb_video` 还依赖 moviepy/wandb 视频链路。

### 8.2 源码审计发现（建议反馈给作者）

1. **噪声注入时间表与论文文字可能不符**：代码 `noise_scale=η·exp(10(t−1))`（t=0≈0，t=1=η），而论文 Eq.(7) 的方差是二次型 `α_t²=(η²−2η)t²+2ηt`（t≈0.53 处最大，σ≈0.23）。两者在 t=1 的端点一致、但中段差别很大；若论文公式确为二次型，则实现与论文存在不一致，建议明确。
2. **β 更新符号**：代码 `new_beta = max(0, beta - 0.1*(-d - H)) = beta + 0.1(H + d)`（熵越高 β 越大→更偏向利用高 Q 候选）；而论文 Eq.(9) 印刷为 `ξ − α[H − H̄]` 方向相反。以代码为准时需要作者澄清。
3. **`to_eval` 的全局 argmax**：`agents/fino.py:188-190` 的 `jnp.argmax(q)` 无 axis；`sample_actions` 若以 batch>1 且 `temperature=0` 调用会产生跨状态污染（官方 `evaluate()` 逐条调用 B=1 不受影响，但调用者容易踩坑）。
4. **CsvLogger 丢列**：`utils/log_utils.py:CsvLogger.log` 在第一次写入时固定 header，之后新增的键（如 FINO 在线阶段的 `evaluation/entropy`、`evaluation/beta`）不会写入 CSV，只在 wandb 中可见；这会妨碍离线分析，建议改为动态表头。
5. **β 不随 checkpoint 保存**：`utils/flax_utils.py:save_agent` 只保存 network params，`beta` 是 `PyTreeNode` 上的标量（初始 10），断点恢复后 β 会重置，可能影响续训行为一致性（`restore_agent` 不改 β）。
6. **`update_entropy` 的随机性**：`main.py:221` 用固定 `jax.random.PRNGKey(FLAGS.seed)`，每次评估都用同一批噪声估计熵，估计方差偏大且不随训练变化；另外 `update_entropy` 内部每次调用都会重新 `@jax.jit` 编译 `one_chunk`。
7. **save/restore 依赖 cwd**：`utils/flax_utils.py` 用 `os.path.join(os.getcwd(), save_dir, ...)`，必须在 `FINO/` 目录启动；且总步数 < `save_interval` 时不会保存 checkpoint（默认 1e6）。
8. **同 env+seed 重跑会覆盖** `exp/<AGENT>/<env>/<seed>` 下的 CSV（`main.py:63`），复现时建议不同 seed 或备份。

---

## 9. 如何在 Linux + GPU 上全量复现（推荐路径）

1. **环境**：Linux（Ubuntu 22.04）+ NVIDIA 驱动 + CUDA 12.x；Python 3.9；直接 `pip install -r requirements.txt`（`jax[cuda12]` 会装 Linux CUDA 插件；`d4rl`/`mujoco-py` 在 Linux 上按官方 README 安装，需要 MuJoCo 2.1 二进制与 `MUJOCO_PY_MUJOCO_PATH`）。
2. **数据**：OGBench 首次运行自动下载到 `~/.ogbench/data`（约 20–300 MB/数据集）；D4RL 数据首次运行自动下载到 `~/.d4rl`。
3. **运行**：README 中的命令可直接逐条执行，例如

   ```bash
   # OGBench 默认任务（7 条）
   python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.alpha=100 --agent.discount=0.995
   python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min
   # D4RL（10 条）
   python main.py --env_name=antmaze-umaze-v2 --agent.alpha=10
   ```

   训练 1.5M 步/任务；建议 `--eval_interval=50000 --eval_episodes=50`（默认值），并保留 wandb 在线模式。
4. **成本估计**：以单张 A100/A5000 上 1M 步 FQL 类算法约 2–5 h 估算（OGBench 官方说明），FINO 因多一个 actor 网络与 10 步 flow 约 1.5–2×，即 **约 3–8 GPU 小时/任务**；55 个任务 ≈ **200–450 GPU 小时**（可用多卡/多机并行）。
5. **验收**：每个任务的 `eval.csv` 最后一行 `evaluation/success`（OGBench）或 `normalized_return`（D4RL）与论文表格对比；若要做消融，用 `--agent.noise_scale=0`（关噪声注入）与减少 `flow_steps` 等。

---

## 10. 交付物清单

| 路径 | 内容 |
|---|---|
| `E:\FINO\FINO` | 官方源码 + 4 处 Windows/CPU 补丁（算法未改） |
| `E:\FINO\_extract\FINO-main` | 未改动的官方源码备份（用于 diff/对照） |
| `docs/01_code_analysis.md` | 724 行源码技术分析（方法-公式-代码映射、baseline 对比、16 条复现坑、README 超参表） |
| `docs/02_reproduction_report.md` | 本报告 |
| `docs/patches.diff` | 全部补丁 diff |
| `constraints.txt` | 钉版依赖 |
| `scripts/bench.py` | CPU 速度基准 |
| `scripts/dl_parallel.py` | OGBench 数据集多线程 Range 下载器（限速服务器专用） |
| `scripts/plot_results.py` | 从 CSV 生成曲线与汇总表 |
| `scripts/dataset_stats.py` | 数据集统计 |
| `experiments/toy/` | 2D 机制验证（脚本、数据、指标、8 张图、RESULTS.md） |
| `exp/FINO/...` | 各运行的 `train.csv/eval.csv/flags.json/checkpoint` |
| `results/` | 训练/评估曲线 PNG、`summary.csv`、`summary.md`、`dataset_stats.json` |
| `logs/` | 安装、下载、训练、恢复测试的完整日志 |

---

## 11. 参考

- 官方仓库：<https://github.com/CTID282/FINO>
- 论文（OpenReview）：<https://openreview.net/forum?id=6wd38R8L0Z>
- OGBench：<https://github.com/seohongpark/ogbench>（数据下载：<https://rail.eecs.berkeley.edu/datasets/ogbench>）
- FQL（本仓库的代码母体）：<https://github.com/seohongpark/fql>
- JAX 安装说明（Linux CUDA 插件 / Windows CPU）：<https://jax.readthedocs.io/en/latest/installation.html>
