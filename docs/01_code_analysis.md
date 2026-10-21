# FINO 源码与技术分析（ICLR 2026 复现报告技术底稿）

> **分析对象**：`https://github.com/CTID282/FINO` 官方实现（JAX），本地路径 `E:\FINO\FINO`
> **论文**：*Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning*（ICLR 2026，OpenReview id `6wd38R8L0Z`，arXiv `2602.18117`）
> **分析方式**：仅静态阅读源码（未运行训练、未修改 `E:\FINO\FINO` 与 `E:\FINO\.venv`），所有结论均标注本地文件与行号；仓库主目录记为 `FINO/`。
> **文档定位**：为复现报告提供“方法—代码—超参—坑”四层可核对的底稿；第 2、6 章可直接改写成报告的方法与实验设置章节，第 5 章可改写为“复现过程中的问题与修正”。

---

## 0. 摘要（TL;DR）

FINO 在实现上就是 **FQL（Flow Q-learning）的超集**（README 第 116 行也明确说明基于 FQL 代码库），在 FQL 的“BC flow 教师 + 一步蒸馏策略 + 双 Q”骨架上加了两处改动：

| 改动 | 论文位置 | 代码位置 | 一句话说明 |
|---|---|---|---|
| **Noise Injection**（噪声注入 flow matching） | Eq.(7)，§4.1 | `agents/fino.py:56-58` + `noise_scale` 配置 | 在插值路径点 `x_t` 上按时间表 `α_t` 加高斯噪声后再送入向量场，使流模型在 `t=1`（数据端）的边际方差变为 `η²`，从而把策略支持集扩大到数据集之外，为在线探索提供“可靠的多走一点”的动作。 |
| **Q-guided 多噪声采样 + 熵自适应 β** | Eq.(8)(9)，§4.2 | `agents/fino.py:154-197`（采样）、`215-232`（β 更新）、`310-335`（GMM 熵估计）、`main.py:220-224`（触发） | 在线阶段对同一状态采多个噪声候选动作，用 `softmax(β·Q)` 而不是 argmax 选动作；β 由 GMM 估计的策略熵反馈调节，自动在探索/利用之间切换。 |

**关键结论速览**（详见正文）：

1. 单步训练时三个损失在**同一次 Adam 更新**里联合优化：`L = L_critic + L_actor`，`L_actor = L_bc_flow + α·L_distill + L_q`（`fino.py:78,109`）。
2. BC flow 与一步策略共用同一 `ActorVectorField` 结构，但**教师带时间输入**、学生不带时间（`fino.py:288-289`）。
3. **噪声注入的实际时间表与“在 t→0 加噪”的常见描述相反**：代码 `noise_scale = exp(-(t*-10+10))·0.1 = 0.1·exp(10(t−1))`（`fino.py:57`），在 `t=0` 处 ≈`4.5e-6`（≈0），在 `t=1` 处 =`0.1`（=η）。这与论文 §4.1 的边界条件 `α_0²=0, α_1²=η²` 一致，而与“在 t→0 处加噪”的表述不符；`t=0` 的“带噪起点”指的是基底噪声 `x_0~N(0,I)`（`fino.py:50`），不是注入噪声。详见 §2.3 的“订正”。
4. 蒸馏损失的目标由**当前 BC flow 教师做 10 步 Euler 积分**得到，且对教师无梯度（`fino.py:65,199-213`）；`α` 只乘蒸馏项，控制“贴教师（保守）”与“被 Q 拉走（激进）”的权衡（README 中 α 从 3 调到 10000）。
5. actor 的 Q 项对 **critic 停止梯度**（`fino.py:70` 未传 `params`），且用 **Q 集成均值**而非 min；`q_agg=min` 只作用于 TD 目标和候选选择（`fino.py:29-32,176-179`）。
6. 在线交互时 `sample_actions(temperature=1)` 走随机 softmax 采样（`fino.py:181-186`），评估时 `temperature=0` 走 `argmax`（`fino.py:188-194`，由 `utils/evaluation.py:44,74` 传入 0）——**β 只影响在线数据采集，不影响最终评估动作**。
7. β 不是网络参数，初始值 10（`fino.py:307`），**不被 checkpoint 保存**（`utils/flax_utils.py:171-188` 只存 network params），断点续训后 β 会重置为 10。
8. `main.py:22` 硬编码 `MUJOCO_GL=egl`，且 `setup_wandb` 默认在线模式（`utils/log_utils.py:68`），是 Windows + 离线复现首先要处理的两个外部依赖点。

---

## 1. 仓库结构与入口

### 1.1 文件清单与职责

仓库只有 3 个子包，代码量很小，属于典型的“单文件算法 + 公共 RL 工具”结构：

| 路径 | 行数 | 职责 |
|---|---:|---|
| `main.py` | 238 | 唯一入口：解析 absl flags + ml_collections 配置、建环境/数据集、训练循环（离线→在线）、日志、评估、保存 |
| `agents/fino.py` | 360 | **FINO 全部算法**：`FINOAgent`（critic/actor 损失、Q-guided 采样、熵估计与 β 更新） |
| `agents/fql.py` | 270 | FQL baseline（FINO 的直接母体） |
| `agents/ifql.py` | 250 | IFQL（IDQL 的 flow 版本） |
| `agents/iql.py` | 261 | IQL（AWR / DDPG+BC 两种 actor loss） |
| `agents/rebrac.py` | 244 | ReBRAC（TD3+BC 变体） |
| `agents/sac.py` | 224 | SAC / RLPD |
| `agents/__init__.py` | 15 | `agents` 注册表：`dict(fql=..., ifql=..., iql=..., rebrac=..., sac=..., fino=...)`，`main.py:104` 按 `config['agent_name']` 取类 |
| `utils/networks.py` | 234 | `MLP`、`Value`（集成 Q/V）、`Actor`（高斯）、`ActorVectorField`（flow 向量场）、`LogParam` |
| `utils/flax_utils.py` | 210 | `ModuleDict`（多网络共享参数容器）、`TrainState`、梯度统计、Orbax 保存/恢复 |
| `utils/datasets.py` | 173 | `Dataset`（冻结的 FrozenDict）与 `ReplayBuffer`（环形缓冲）、frame stack、图像增广 |
| `utils/encoders.py` | 108 | IMPALA ResNet 视觉编码器及注册表 |
| `utils/evaluation.py` | 105 | `evaluate()`：跑 episode、统计 info、可选出视频 |
| `utils/log_utils.py` | 148 | `CsvLogger`、`setup_wandb`、wandb 视频拼图 |
| `envs/env_utils.py` | 155 | `make_env_and_datasets()`：按 env 名分派 OGBench / D4RL(antmaze/adroit/locomotion)；`EpisodeMonitor`、`FrameStackWrapper` |
| `envs/d4rl_utils.py` | 58 | D4RL 环境构造与数据集转换（terminals/masks/reward 平移） |
| `README.md` | 132 | 复现命令（OGBench 45 任务 + D4RL 10 任务）与超参 |
| `requirements.txt` | 18 | 依赖与版本钉死 |

### 1.2 `main.py` 执行流程（按行号）

1. **入口与全局设置**：`main.py:22` 在 import 后立即 `os.environ["MUJOCO_GL"] = "egl"`；`:24-49` 定义 absl flags，其中 `:49` 用 `config_flags.DEFINE_config_file('agent', 'agents/fino.py', lock_config=False)` 把 agent 超参绑到一个 Python 配置文件，默认就是 `agents/fino.py`。
2. **实验命名与 wandb**：`:54` `get_exp_name(seed)`，`:55-56` 用 `"{agent_name}_{seed}"` 覆盖（时间戳实际被丢弃），`:57-61` `setup_wandb(project='FINO')`。
3. **保存目录与 flags 快照**：`:63-67` 目录为 `exp/{AGENT}/{env_name}/{seed}`，并把所有 flags（含 agent config）写入 `flags.json`。
4. **环境与数据集**：`:71` `make_env_and_datasets(...)` 返回 `(env, eval_env, train_dataset, val_dataset)`；`:72-75` 断言：视频只支持 OGBench singletask，在线微调不支持 visual 环境。
5. **Dataset/ReplayBuffer 二选一**：`:82-92` 默认（`balanced_sampling=0`）把离线数据集直接扩展成 `ReplayBuffer`（容量 `max(1500000, N+1)`），在线数据直接写回同一个 buffer；`balanced_sampling=1` 时另建 buffer，在线时按 1:1 采样（`:173-177`）。
6. **建 agent**：`:102` 先 `sample(1)` 拿 example batch；`:104-110` `agents[config['agent_name']].create(seed, obs, act, config)`。
7. **主循环**：`:126` 从 `i=1` 到 `offline_steps + online_steps`。
   - **离线**：`:127-134` 从 dataset 采 `batch_size=256`，调 `agent.update(batch)`；ReBRAC 额外传 `full_update=(i % actor_freq == 0)`。
   - **在线**（`i > offline_steps`）：`:137-143` 用独立 `online_rng` 生成动作，**`temperature=1`**；`:146` `env.step`；`:149-153` 仅对 D4RL antmaze（umaze/diverse/play）把 reward 减 1；`:155-164` 写入 replay buffer；`:173-179` 采样训练 batch；`:181-184` 更新。
8. **训练日志**：`:187-198` 每 `log_interval=5000` 步记录 update_info（含 `grad/*`）、可选验证集损失（`:189-192`，仅 OGBench 有 val_dataset）、时间、`exploration/*`。
9. **评估**：`:200-227` 每 `eval_interval=50000` 步（以及 `i==1`）调 `utils/evaluation.py:evaluate`，`eval_episodes=50`；**FINO 额外**在 `i > offline_steps` 时调 `agent.update_entropy(...)` 并记录 `evaluation/entropy`、`evaluation/beta`（`:220-224`）。
10. **保存**：`:229-231` 每 `save_interval=1e6` 步 `save_agent`；`:233-234` 关闭 CSV logger。

### 1.3 配置系统（ml_collections + absl）

- 每个 agent 文件提供 `get_config() -> ml_collections.ConfigDict`（如 `fino.py:338-360`），其中 `ob_dims`、`action_dim`、`encoder` 是 `config_dict.placeholder(...)`，在 `create()` 里回填（`fino.py:305-306`）。
- `main.py:49` 的 `DEFINE_config_file(..., lock_config=False)` 允许命令行用点号覆盖，例如 README 中 `--agent.alpha=100 --agent.q_agg=min --agent.discount=0.995`。
- 运行结束后 `flags.json` 保存了完整配置（`main.py:65-67`，`utils/log_utils.py:53-59` 把 ConfigDict 转 dict），可用于复现报告记录超参。
- 注意：配置文件的默认路径是**相对路径** `agents/fino.py`，因此必须从 `FINO/` 目录启动；`--agent=agents/fql.py` 可切换 baseline。

### 1.4 一次运行的数据流

```
                   ┌──────────────────────────── main.py ────────────────────────────┐
env_name ──► make_env_and_datasets (envs/env_utils.py:91)
                   │   ├─ OGBench : ogbench.make_env_and_datasets → Dataset.create (冻结数组)
                   │   └─ D4RL    : d4rl_utils.make_env / get_dataset (terminals/masks 重算)
                   ▼
            train_dataset / replay_buffer (utils/datasets.py:34,113)
                   │
                   ▼
  config = FLAGS.agent ──► agent = FINOAgent.create (agents/fino.py:234)
                   │            ├─ critic + target_critic (Value, 2-ensemble)
                   │            ├─ actor_bc_flow (ActorVectorField, 带 t)
                   │            └─ actor_onestep_flow (ActorVectorField, 不带 t)
                   ▼
  for i in 1..(offline+online):
      offline: batch = dataset.sample(256) ──► agent.update(batch) [jit]
      online : a = sample_actions(s, temperature=1) → env.step → buffer.add_transition → update
      log   : update_info ──► wandb + train.csv
      eval  : evaluate(sample_actions(temperature=0)) ──► wandb + eval.csv
              (FINO) update_entropy → β
      save  : save_agent → orbax 每 1e6 步
```

---

## 2. FINO 方法拆解（论文 ↔ 代码）

### 2.1 总览：三个可训练网络 + 两个训练阶段

`FINOAgent.create`（`fino.py:234-307`）构造如下网络（均由 `utils/flax_utils.py:18-52` 的 `ModuleDict` 容器管理，**共享同一个 Adam 优化器** `fino.py:298`）：

| 模块名 | 类型 | 输入 | 初始化参数 | 作用 |
|---|---|---|---|---|
| `critic` | `Value(num_ensembles=2)`（`fino.py:266-271`） | `(s, a)` | `ex_observations, ex_actions` | 双 Q 集成 |
| `target_critic` | `copy.deepcopy(critic)`（`:287`） | 同上 | 同上 | TD 目标；参数初始化为 critic 的拷贝（`:302-303`） |
| `actor_bc_flow` | `ActorVectorField`（`:272-277`） | `(s, a_t, t)` | `(ex_obs, ex_act, ex_times)`，`ex_times = ex_actions[..., :1]`（`:253`） | 多步 flow 教师（behavior flow） |
| `actor_onestep_flow` | `ActorVectorField`（`:278-283`） | `(s, z)`（`times=None`） | `(ex_obs, ex_actions)` | 一步策略 π_ω（与环境交互） |

`FINOAgent` 是 `flax.struct.PyTreeNode`，字段为 `rng / network / config / beta`（`fino.py:16-21`）；`beta` 是普通浮点字段（不是可训练参数），用于在线采样温度（见 §2.7）。

**论文 §4.3 明确以 FQL 为 backbone**，因此 `fino.py` 与 `fql.py` 的 critic、蒸馏、Q 项几乎逐行相同，差别仅在噪声注入与采样/β（见 §4.3 的逐文件 diff 总结）。

### 2.2 BC flow 损失（论文 Eq.4 / Eq.7 的落地）

`fino.py:46-61`：

```python
x_0 = jax.random.normal(x_rng, (B, A))        # 第 50 行：基底噪声
x_1 = batch['actions']                         # 第 51 行：数据动作
t   = jax.random.uniform(t_rng, (B, 1))        # 第 52 行：t ~ U[0,1]
x_t = (1 - t) * x_0 + t * x_1                  # 第 53 行：OT 线性插值路径
vel = x_1 - x_0                                # 第 54 行：回归目标（常数速度）
...
pred = self.network.select('actor_bc_flow')(batch['observations'], normal_samples, t, params=grad_params)  # 第 60 行
bc_flow_loss = jnp.mean((pred - vel) ** 2)     # 第 61 行
```

对应论文 Eq.(4) `L_π(θ)=E||v_θ(t,s,x_t)-(x_1-x_0)||²`（`t~U[0,1]`）。注意三点实现细节：

1. 代码的回归目标是 `x_1 - x_0`，而不是 Eq.(7) 一般形式里的 `x_1-(1-η)x_0`；论文 §4.3 说明这是刻意的实现简化（“simply use the target vector x_1−x_0”）。
2. `x_t` 只用于构造噪声注入点，真正送入网络的是 `normal_samples`（第 58 行）——这是 FINO 与 FQL 唯一的 loss 级差异。
3. 向量场网络把 `[s, a, t]` 拼接输入（`utils/networks.py:216-232`），`t` 是标量（每样本 1 维）。

### 2.3 Noise Injection（论文 Eq.7 / §4.1）——实现细节与一处表述订正

#### 2.3.1 代码事实

`fino.py:56-58`：

```python
normal_noise = jax.random.normal(nor_rng, shape=(B, A))          # ε ~ N(0, I)
noise_scale  = jnp.exp(-(t * -10 + 10)) * self.config['noise_scale']  # = η·exp(10(t−1))
normal_samples = x_t + noise_scale * normal_noise                # 注入后的网络输入
```

其中 `self.config['noise_scale']` 默认 `0.1`（`fino.py:357`），即论文的 `η=0.1`。因此注入噪声标准差为

```
α_t = η · exp(10(t − 1)) = η · exp(−10(1 − t))
```

数值（η=0.1）：

| t | 0 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.99 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| α_t | 4.5e-6 | 1.2e-5 | 5.5e-5 | 6.7e-4 | 8.2e-3 | 3.7e-2 | 8.2e-2 | **0.1** |

即：**注入强度在 t=0 处≈0，在 t=1 处达到 η=0.1 的最大值；中间点极小**。这是一个“平移指数型”schedule（论文 §4.3 称 “smooth shifted exponential schedule”，“满足 α_0²≈0 与 α_1²=η²”）。

论文 Eq.(7) 的**理论方差表**为 `α_t² = (η²−2η)t² + 2ηt`（`η=0.1` 时 `α_t² = −0.19t²+0.2t`，在 `t≈0.526` 取最大 `σ≈0.229`），与代码实现并不相同；论文 §4.3 明确说实现改用指数 schedule 但**保留相同的边界条件**。复现报告应把“理论 schedule vs 实现 schedule”作为可做消融的点（见 §7）。

#### 2.3.2 论文动机（逐条）

1. **问题根源**：FQL 用 `σ_min=0` 的标准 flow matching，条件概率路径在 `t=1` 收缩成单个数据点（Eq.(6) 方差 `(1−(1−σ_min)t)²=0`），因此策略支持集几乎完全局限在数据集内。论文 §3 用 `antmaze-giant` 的 FQL 轨迹（只走数据集里出现过的上侧路线）说明这一点。
2. **扩support**：在路径点 `x_t` 上加噪声后再回归 `x_1−x_0`，等价于让条件路径在 `t=1` 保留方差 `η²`（Proposition 1：`p_t^FINO(x|x_i)=N(tx_i,(1−(1−η)t)²I)`），从而 **`Var(X_t^FINO) ≥ Var(X_t^FM)`（Theorem 2）**：流模型在数据端生成的动作分布更宽、且仍围绕数据点（论文 Fig.2 toy：四个模态周围出现一圈扩大但仍有界的覆盖）。
3. **为什么噪声加在 t=1 端（数据端）而不是 t=0 端**：这是该设计的关键。`t=0` 处本来就是高斯基底噪声，往噪声上加噪声不会扩大动作支持；只有把 `t=1` 端从“点”变成“方差 η² 的球”，才能让生成的**动作**覆盖比数据集更宽的区域。代码 `α_0≈0, α_1=η` 正是这一逻辑；训练时网络看到 `x_1+ηε` 却仍被要求输出 `x_1−x_0`，等价于学习一个把扰动点“拉回”路径的收缩场，最终使边际分布变宽（论文用连续方程 / Theorem 1 证明仍是合法的 CNF）。
4. **与一步蒸馏的关系**：一步策略 π_ω 的蒸馏目标 `a_θ(s,z)` 由该“被加宽的教师”积分得到（§2.4），因此 π_ω 会自然覆盖更宽的动作集；随后 Q 项（§2.5.3）在这个更宽的区域里做最大化，就得到“数据集中没有但合理”的探索动作。**因此“noise injection → 更宽的教师分布 → 更宽的一步策略 → 在线探索”是论文的因果链**。

> **⚠ 与任务描述的一处订正（重要）**：任务/部分解读中“noise injection 在 t→0 处加噪，使得蒸馏一步策略能从带噪起点学习”这一说法与官方代码/论文不一致。
> - 代码 `fino.py:57` 给出 `α_t = η·exp(10(t−1))`，**t=0 时 α_t≈4.5e-6≈0**，噪声在 **t→1** 才增大到 η；
> - 论文 §4.1 明确 `α_0²=0, α_1²=η²>0`；
> - 一步策略的“带噪起点”是基底噪声 `x_0~N(0,I)`（`fino.py:50,64,142`），它不是注入噪声，而是 flow 的初始分布；蒸馏时教师本身就是“从该噪声出发积分”的，不需要额外在 t=0 加噪。
>
> 因此报告里建议写成：**“在概率路径的终端（数据端 t→1）注入受控噪声，扩大流模型在 t=1 的边际方差，再通过蒸馏把扩大的覆盖迁移到一步策略”**。若报告需要引用“加噪起点”的说法，应明确它指基底噪声 `x_0`，而非 `ε_t` 注入项。

#### 2.3.3 噪声注入对训练目标的影响（一句话版）

BC flow 的回归目标不变（`vel = x_1 - x_0`），但**输入分布被改变**：网络必须在 `x_t` 的 η-邻域内都预测同一个速度。这相当于对向量场做局部平滑/收缩正则，使学到的 marginal 在 `t=1` 有额外方差（论文 Prop.1 / Thm.2），最终扩大策略 support。

### 2.4 One-step Distillation（论文 Eq.5 的第二项）

`fino.py:63-67`：

```python
rng, noise_rng = jax.random.split(rng)
noises = jax.random.normal(noise_rng, (B, A))                          # z ~ N(0, I)
target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)  # 教师积分，无梯度
actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
```

教师积分 `compute_flow_actions`（`fino.py:199-213`）：

```python
if encoder: observations = self.network.select('actor_bc_flow_encoder')(observations)   # :205-206
actions = noises                                                                        # :207
for i in range(flow_steps):                                                             # :208
    t = jnp.full((..., 1), i / flow_steps)                                              # :209
    vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)  # :210
    actions = actions + vels / flow_steps                                               # :211
actions = jnp.clip(actions, -1, 1)                                                      # :212
```

要点：

- `flow_steps` 默认 10（`fino.py:354`），Euler 步长 0.1，`t` 取 `0,0.1,...,0.9`（**不含 1**），因此教师输出是 `x_1` 的近似（10 步 Euler 误差是复现时值得记录的指标之一）。
- 教师参数在蒸馏目标里**不流梯度**：`compute_flow_actions` 没有传 `params`，`TrainState.__call__` 在 `params=None` 时使用存储参数并停止梯度（`utils/flax_utils.py:92-116`）；因此蒸馏只更新 `actor_onestep_flow`，不会把“教师追学生”的耦合引入。
- 蒸馏目标每次 update 都重新算一遍 10 步 Euler（batch 256 × 10 次前向），是 FINO 相对普通 DDPG 类方法的主要计算开销之一。
- 这与 FQL 的实现完全一致（`fql.py:62-66, 156-171`），说明蒸馏本身不是 FINO 的贡献；FINO 的贡献是**教师的分布被噪声注入加宽**，从而蒸馏出的 π_ω 也变宽。

### 2.5 Actor 总损失：`L_actor = L_bc_flow + α·L_distill + L_Q` 逐项解释

`fino.py:69-78`：

```python
actor_actions = jnp.clip(actor_actions, -1, 1)                    # :69
qs = self.network.select('critic')(batch['observations'], actions=actor_actions)  # :70（无 params → critic 不回传梯度）
q  = jnp.mean(qs, axis=0)                                        # :71（集成均值）
q_loss = -q.mean()                                               # :73
if self.config['normalize_q_loss']:                             # :74（默认 False）
    lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())           # :75
    q_loss = lam * q_loss                                        # :76
actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss  # :78
```

| 项 | 代码 | 论文 | 作用/动机 | 权重与调参 |
|---|---|---|---|---|
| `bc_flow_loss` | `:50-61` | Eq.(4)+Eq.(7) | 让多步 flow 教师拟合数据动作的 OT 向量场，保留多模态建模能力；噪声注入后其 support 比数据集更宽，是“探索多样性”的来源 | 权重恒为 1 |
| `α·distill_loss` | `:63-67,78` | Eq.(5) 第二项 `α||a_ω−a_θ||²` | 把一步策略 π_ω 锚定在多步教师上：教师是多模态、被加宽的“行为分布”，蒸馏保证 π_ω 继承该分布，而不是被 Q 拉成退化解。α 越大越“行为保守/贴教师”，越小越“Q 激进” | `alpha` 默认 10（`:353`）；README 按环境在 3~10000 调（见 §6） |
| `q_loss` | `:69-73` | Eq.(5) 第一项 `−Q(s,a_ω)` | 在教师覆盖的宽 action 区域内做 DDPG 式 Q 最大化，把 π_ω 推向高价值动作；`critic` 不回传梯度（`:70` 未传 `params`），Q 只作为可微的标量函数对 `actor_actions` 求导 | 权重恒为 1；`normalize_q_loss` 默认关闭，因此 reward 的尺度会直接决定 Q 项与 BC 项的平衡 |

**三者关系**：`L_bc_flow` 训练教师（决定“可以探索到哪里”），`α·L_distill` 把教师的知识搬到可高效推理的一步策略，`L_Q` 在教师允许的范围内挑高价值动作。FQL 的 Eq.(5) 只有后两项；FINO 的差别在于第一项（教师）被噪声注入改变了分布，从而第二项的锚点变宽。

**与论文的对应关系**：论文 Eq.(5) 写成 `−Q + α||a_ω−a_θ||²`；代码把“教师自身训练项” `L_bc_flow` 也放进同一个 actor 总损失，所以是三项相加。这是 FQL 代码库的既有写法（`fql.py:79` 完全相同），并非 FINO 新增。

**关于 β（熵自适应）**：β 不在 actor loss 里，而是采样/在线交互阶段的温度（§2.7）；论文把它归入“entropy-guided sampling”。为完整性在 §2.7 解释它与噪声注入的配合关系。

### 2.6 Critic 与 target critic（TD 学习）

`fino.py:23-44`：

```python
next_actions = self.sample_batch_actions(batch['next_observations'], seed=sample_rng)  # :25（一步策略）
next_actions = jnp.clip(next_actions, -1, 1)                                           # :26
next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)  # :28
next_q = next_qs.min(axis=0) if q_agg == 'min' else next_qs.mean(axis=0)                # :29-32
target_q = batch['rewards'] + discount * batch['masks'] * next_q                        # :34
q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)  # :36
critic_loss = jnp.square(q - target_q).mean()                                           # :37
```

要点：

1. **目标动作来自一步策略**，不是多步 flow 教师，也不是数据集动作。`sample_batch_actions`（`:134-151`）用 `actor_onestep_flow(s, z)`，`z~N(0,I)` 后 clip；它与在线交互的 `sample_actions` 不同（后者带 Q 选择与 β），因此 TD 目标计算便宜且确定。
2. **`q_agg` 语义**：只决定 `target_q` 的集成聚合（`min`=保守，`mean`=中性），默认 `mean`（`:352`）；README 在 antmaze 系列改用 `min`（§6）。
3. **TD 目标用 `masks` 而不是 `terminals`**（`:34`）：`masks` 来自数据集（D4RL 在 `d4rl_utils.py:28-48` 重算），对 time-limit 截断保留 bootstrap；在线新增 transition 时 `masks=1.0-terminated`（`main.py:161`），即 truncated 仍 bootstrap。
4. **critic 是 2 集成**（`fino.py:266-271`），`target_critic` 用 `tau=0.005` 软更新（`:112-119`、`:130`），只在 critic 上做 target update，actor 没有 target（与 FQL 一致）。
5. `critic_loss` 里没有显式 BC/保守项，FINO 的“保守性”由 `α·distill` 提供（FQL 论文思路）。

### 2.7 Q-guided 多噪声采样 + 熵自适应 β（论文 Eq.8 / Eq.9）

#### 2.7.1 采样：`sample_actions`（`fino.py:153-197`）

```python
num_samples = min(10, (action_dim + 1) // 2)                                   # :163
n_noises = jax.random.normal(action_seed, (num_samples, *obs_batch, A))        # :164-171
n_observations = jnp.broadcast_to(observations, (num_samples,) + observations.shape)  # :172
actions = actor_onestep_flow(n_observations, n_noises); clip                   # :173-174
q = critic(n_observations, actions=actions)                                    # :175
q = q.min(axis=0) if q_agg == 'min' else q.mean(axis=0)                        # :176-179

def to_env(inputs):                                                            # :181
    logits = (q / jnp.abs(q).mean()) * self.beta                               # :182
    idx = jax.random.categorical(sample_seed, logits=logits, axis=0)           # :183
    ...  # 按 idx 从候选动作中取（:184-186）

def to_eval(inputs):                                                           # :188
    action = actions[jnp.argmax(q)]                                            # :189

action = jax.lax.cond(temperature == 0, to_eval, to_env, ())                   # :192-194
action = action.squeeze()                                                      # :195
```

- **候选数**：`K = min(10, (A+1)//2)`，与论文“候选数为动作维度一半”一致（对 2 维动作 K=1，此时退化为单动作、与 FQL 行为相同；对 D4RL/OGBench 常见 5~24 维动作 K=3~10）。
- **在线（temperature=1，`main.py:143`）**：`p(k) ∝ exp(β·q_k/mean|q|)` 的 categorical 采样，**允许选到低 Q 动作**，这就是在线探索的直接来源；论文 Eq.(8) 即 `p(i)=exp(ξQ(s,a_i))/Σexp(ξQ(s,a_j))`，代码用 `mean|q|` 做归一化（数值稳定 + 与 `normalize_q_loss` 思路一致）。
- **评估（temperature=0，`evaluation.py:44,74`）**：`argmax` 选最高 Q 候选，**确定性**，与论文“inference 时贪心选 Q 最高动作”一致。
- 注意：β 只出现在 `to_env`，所以**β 只影响在线数据采集分布，不影响 eval 指标的动作选择**。

#### 2.7.2 β 的熵自适应更新

触发位置：`main.py:220-224`——仅在 `i > offline_steps` 的**评估时刻**（eval 每 50k 步）调用：

```python
if hasattr(agent, 'update_entropy') and (i > FLAGS.offline_steps):
    agent, entropy = agent.update_entropy(batch['observations'], jax.random.PRNGKey(FLAGS.seed))
    eval_metrics['evaluation/entropy'] = entropy
    if hasattr(agent, 'beta'):
        eval_metrics['evaluation/beta'] = agent.beta
```

实现（`fino.py:215-232`）：

```python
def update_entropy(self, observations, seed, n_samples=200, n_components=3):
    sample_keys = jax.random.split(seed, n_samples)
    @jax.jit
    def one_chunk(chunk_keys):
        return jax.vmap(lambda k: self.sample_actions(observations, seed=k))(chunk_keys)
    chunks = []
    for i in range(0, n_samples, 25):
        chunks.append(np.asarray(one_chunk(sample_keys[i:i+25])))
    action_samples_np = np.concatenate(chunks, axis=0)          # (n_samples, B, A)
    action_samples_np = action_samples_np.transpose(1, 0, 2)   # (B, n_samples, A)
    entropy = estimate_entropy_sklearn(action_samples_np, num_components=n_components)
    new_beta = max(0, self.beta - 0.1 * (-self.config['action_dim'] - entropy))
    return self.replace(beta=new_beta), entropy
```

- **熵估计**：`estimate_entropy_sklearn`（`fino.py:310-335`）对每个 state 的 200 个动作样本拟合 **3 分量 full-covariance GMM**，用 `Σ_j w_j·H_j + H(w)`（混合权重熵 + 各分量高斯微分熵）近似策略熵。这正是论文 §4.3 的做法（一步策略无解析密度，故用 GMM 估计）。
- **β 更新式**：代码为 `β_new = max(0, β − 0.1·(−d_A − H)) = max(0, β + 0.1·(H + d_A))`。若把目标熵记作 `H̄ = −d_A`（与 SAC 的 `target_entropy=−0.5d` 同类，但这里用满 `−d`），等价于 `β_new = β + 0.1·(H − H̄)`：**策略熵 H 越大 → β 越大 → softmax 越尖 → 越偏 exploitation；H 越小 → β 越小 → 越平 → 越偏 exploration**。这与论文 §4.2 的定性描述（以及常见解读“熵高→利用、熵低→探索”）一致，形成负反馈把熵拉向 `−d_A`。
- **⚠ 与论文 Eq.(9) 印刷形式的差异（需在报告中注明）**：`ar5iv` 文本给出的 Eq.(9) 是 `ξ_new = ξ − α_ξ[H − H̄]`；若按字面，H 增大时 ξ 应变小（更平缓=更多探索），与代码方向相反。代码实际是 `ξ + α_ξ(H − H̄)`。考虑到论文正文对减小/增大 ξ 的解释、以及 §4.3 的实现描述，**更可能的情况是公式排版/符号约定与代码不一致**；复现时建议以代码为准，并做“固定 β / 反向更新”的消融作为报告中的讨论点。
- **目标熵**：代码里没有显式 `H̄` 变量，`−action_dim` 就是 `H̄` 的硬编码（`fino.py:231`）。若严格按论文/其他实现重写，应把 `H̄` 显式化为超参。
- **β 初值**：`create` 里 `beta=10.0`（`fino.py:307`），没有上限截断（只有 `max(0,·)` 下界）；每次更新步长 0.1。
- **采样开销**：每次 β 更新要 200 次 `sample_actions`（每次 B=256 状态 × K 候选，K 为 `min(10,(A+1)//2)`），再拟合 256 个 GMM。这是 eval 阶段的主要额外耗时；`one_chunk` 在函数内重新定义 `@jax.jit`，每次调用都会新建 jitted 函数对象（会重新 trace/编译），属于可优化但不影响正确性的实现细节。
- **随机数固定**：`main.py:221` 传的是固定 `jax.random.PRNGKey(FLAGS.seed)`，每次 β 更新用**同一组**采样 key（只有策略参数在变），因此熵估计的随机方差被固定、不同 eval 之间高度相关；如果报告要分析熵曲线，建议改成每次 split 一个持续 RNG。
- **β 不参与训练、不参与 checkpoint**：`save_agent` 只保存 `agent.network.params`（`utils/flax_utils.py:171-188`），恢复时 `create()` 重新把 β 置 10（`fino.py:307`），因此**离线/在线两阶段之间若用 restore 断点续训，β 的在线自适应历史会丢失**（本次复现未使用 restore，属潜在坑）。

#### 2.7.3 与噪声注入的配合

噪声注入把教师/π_ω 的动作分布加宽（§2.3），于是同一状态下 K 个候选动作的差异更大、Q 值分布更分散；`softmax(βQ)` 才能有意义地在“多个合理动作”间做探索/利用权衡。若没有噪声注入（η=0），π_ω 会退化到接近确定性/数据集行为，多候选几乎重合，β 自适应就失去作用——这解释了论文消融中“w/o Noise Injection”性能显著下降。

### 2.8 offline → online 微调流程（论文 Algorithm 1）

| 阶段 | 论文 | 代码 |
|---|---|---|
| 离线预训练 | 用 Eq.(5)(7)+TD 更新 ω,θ,ϕ | `main.py:127-134`：`batch = train_dataset.sample(256)`，`agent.update(batch)` |
| 在线交互 | 采 N_sample 个候选 → 按 Eq.(8) 选动作 → 环境 step → 存 buffer | `main.py:139-164`：reset 后 `sample_actions(temperature=1)` → `env.step` → `replay_buffer.add_transition` |
| 在线更新 | 用 (s,a,r,s') 继续更新 ω,θ,ϕ | `main.py:173-184`：从（含在线数据的）buffer 采样后 `agent.update` |
| 熵更新 | 每 N_ξ 步估计 H、按 Eq.(9) 更新 ξ | `main.py:220-224`：只在线上阶段、且仅在 eval 时刻（默认每 50k 步）触发 |
| 推理 | 贪心选 Q 最高动作 | `utils/evaluation.py:44,74` 传 `eval_temperature=0` → `fino.py:188-194` argmax |

**关键实现行为**（对复现报告很重要）：

1. **离线/在线的切换靠步数**：`i <= offline_steps` 为离线；之后进入在线，环境交互与 buffer 写入从 `i=offline_steps+1` 开始。默认 1e6 + 5e5（`main.py:33-34`）。
2. **在线 reward 平移**：仅当 `'antmaze' in env_name and ('diverse' or 'play' or 'umaze' in env_name)` 时 `reward -= 1.0`（`main.py:149-153`），与离线数据集里对 D4RL antmaze 的 `rewards -= 1`（`envs/d4rl_utils.py:37`）配合，保证两阶段 reward 定义一致。OGBench 不满足该条件（其名字里通常没有这些子串，且 reward 本身是 shaped），不会被平移。
3. **在线 buffer**：默认 `replay_buffer = train_dataset`（`main.py:89-92`），离线数据与在线数据混在同一个环形 buffer；`balanced_sampling=1` 时才分离并做 1:1 采样（`main.py:173-177`）。
4. **评估不打断训练**：`evaluate` 用独立 `eval_env`（`main.py:71`），不写入 buffer。
5. **在线动作是随机的**：`temperature=1` 使 `to_env` 分支生效；这是论文“用熵引导采样进行探索”的落点。离线阶段没有环境交互，β 不变（保持 10）。

### 2.9 一个 FINO 训练 step 的完整数据流（伪代码）

```
输入 batch: (s, a, s', r, mask)

# ---- critic ----
z' ~ N(0,I); a' = clip(π_ω(s', z'))                     # fino.py:25-26,134-151
Q' = target_critic(s', a')                              # (2, B)
y  = r + γ·mask·agg_q(Q')                               # agg ∈ {min, mean}, fino.py:29-34
L_Q = mean((critic(s,a) - y)²)                          # fino.py:36-37

# ---- actor ----
x0 ~ N(0,I); x1 = a; t ~ U[0,1]
x_t = (1-t)x0 + t x1;  vel = x1 - x0                     # fino.py:50-54
α_t = 0.1·exp(10(t-1));  x̃_t = x_t + α_t·ε, ε~N(0,I)   # fino.py:56-58
L_bc = mean((v_θ(s, x̃_t, t) - vel)²)                    # fino.py:60-61
z ~ N(0,I);  a_θ = Euler_10(v_θ, s, z);  a_ω = π_ω(s,z)
L_distill = mean((a_ω - a_θ)²)                          # fino.py:63-67,199-213
L_q = -mean( critic(s, clip(a_ω)).mean_over_ensemble )  # fino.py:69-73（critic 不回传梯度）
L_actor = L_bc + α·L_distill + L_q                      # fino.py:78

# ---- 更新 ----
L = L_Q + L_actor;  grads = ∇_params L;  Adam step      # fino.py:109,129
target_critic ← τ·critic + (1-τ)·target_critic          # fino.py:112-119,130
```

### 2.10 论文公式—代码映射总表

| 论文 | 内容 | 代码 |
|---|---|---|
| Eq.(4) | flow matching BC 损失 | `fino.py:46-61`（`bc_flow_loss`），`fql.py:46-59` |
| Eq.(5) | 一步策略 = −Q + α·蒸馏 | `fino.py:63-78` |
| Eq.(6) | FM 条件路径（σ_min=0） | 隐含在 `fino.py:50-54` 的直线插值 |
| Eq.(7) | 噪声注入目标 | `fino.py:56-58` + `get_config.noise_scale`（`:357`） |
| Eq.(8) | softmax(ξQ) 候选采样 | `fino.py:181-186`（在线） |
| Eq.(9) | ξ 熵自适应更新 | `fino.py:230-231`（实际符号见 §2.7.2 说明） |
| Prop.1 / Thm.1 / Thm.2 | 噪声注入的合法性/方差扩大 | 无显式代码，对应 `noise_scale=η` 与实现 schedule |
| Algorithm 1 | 离线/在线两阶段 + 熵更新 | `main.py:126-231` |
| §4.3 GMM 熵估计 | `fino.py:310-335` |
| §4.3 候选数≈d/2 | `fino.py:163` |
| §4.3 η=0.1 | `fino.py:357` |

---

## 3. 关键实现细节（支撑复现的基础设施）

### 3.1 `utils/networks.py`

- `MLP`（`:33-60`）：`Dense → gelu → (LayerNorm)` 堆叠，`activate_final` 控制最后一层是否激活；`sow('intermediates','feature')` 在本仓库没有被消费（仅 FQL 系可选辅助）。
- `Value`（`:152-194`）：`num_ensembles>1` 时用 `nn.vmap` 做集成（`:13-23`），输入为 `concat([obs, action])`（`:184-192`），输出 squeeze 后 shape `(num_ensembles, B)`；`value_net` 的最后一维是 1。
- `Actor`（`:81-149`）：高斯策略，`temperature` 直接乘在 `scale_diag` 上（`:145`）→ **temperature=0 时 std=0，采样退化为均值（确定性）**；`tanh_squash` 时用 `TransformedWithMode`（`:74-79`）取 mode。
- `ActorVectorField`（`:197-234`）：`times=None` 时输入 `[obs, action]`（一步策略），否则 `[obs, action, t]`（教师）；`is_encoded=True` 时跳过 encoder（教师积分时先编码一次，避免 10 步重复编码，`:205-210`）。
- `LogParam`（`:63-71`）：SAC 的可学习温度（`alpha`）。

### 3.2 `utils/flax_utils.py`

- `ModuleDict`（`:18-52`）把多个 `nn.Module` 组合进一个 `flax` 模型，参数名形如 `modules_critic`、`modules_target_critic`；`TrainState.select(name)`（`:118-120`）返回 `functools.partial(self, name=name)`。
- **梯度约定**（`:92-116`）：`TrainState.__call__` 的 `params=None` 表示“用存储参数、停止梯度”，传当前 `grad_params` 才回传。FINO 中哪些网络回传梯度完全由是否传 `params` 决定，见 §2.5/§2.6。
- `apply_loss_fn`（`:134-161`）：`jax.grad(loss_fn, has_aux=True)` 后额外统计 `grad/max`、`grad/min`、`grad/norm` 写入日志。
- 所有模块共享一个 `optax.adam(3e-4)`（`fino.py:298`），**没有按模块区分学习率**，也没有梯度裁剪。
- 保存/恢复：`save_agent`（`:171-188`）按参数名拆分 `modules_` 前缀，每个模块单独用 `orbax.checkpoint.PyTreeCheckpointer` 存到 `os.getcwd()/save_dir/epoch/<module>`；`restore_agent`（`:191-210`）对称恢复，并重建 optimizer state（`TrainState.load`，`:163-168`）。注意 **保存路径基于 `os.getcwd()`**，必须从仓库根运行。

### 3.3 `utils/datasets.py`

- `Dataset(FrozenDict)`（`:34-110`）：`create(**fields)` 断言含 `observations` 并冻结数组；`sample`（`:66-90`）支持 `frame_stack`（视觉）与 `p_aug` 随机裁剪增广；`get_subset`（`:92-98`）在 `return_next_actions=True` 时额外给出 `next_actions`（ReBRAC 专用，`:96-97` 注释警告轨迹末尾不正确）。
- `initial_locs`/`terminal_locs`（`:58-60`）由 `terminals>0` 计算，供 frame stack 用。
- `ReplayBuffer`（`:113-173`）：环形 buffer；`create_from_initial_dataset`（`:135-152`）把离线数据放前面，`size=pointer=N`；`add_transition`（`:161-169`）覆盖写、`size=max(pointer,size)`；**没有优先级采样**，均匀随机（`:62-64`）。
- 在线阶段 `balanced_sampling=0` 时离线与在线数据共享同一 buffer，因此离线数据会被逐渐覆盖（buffer 容量 1.5M > 1M 离线 + 0.5M 在线默认总和，默认恰好不会覆盖，但改变步数时要注意）。

### 3.4 `envs/env_utils.py` 与 `envs/d4rl_utils.py`

- `make_env_and_datasets`（`env_utils.py:91-155`）分派：
  - `'singletask' in env_name` → OGBench（`:103-110`），有 `val_dataset`；`ogbench.make_env_and_datasets` 负责下载数据集（首次运行需要网络）。
  - D4RL antmaze（`umaze/diverse/play`，`:111-118`）、D4RL Adroit（`pen/hammer/relocate/door`，`:119-127`）、D4RL locomotion（`:128-134`）。
  - 最后统一 `action_clip_eps=1e-5` 把数据动作裁到 `[-1+ε, 1-ε]`（`:145-153`）。
- `EpisodeMonitor`（`:13-57`）：在 episode 结束时把 `return/length/duration` 写进 `info['episode']`，并过滤 `privileged/proprio` 键（`:107-108` 传入）；若 env 有 `get_normalized_score`（D4RL）则计算 `normalized_return`（`:48-51`）。
- D4RL 数据转换（`d4rl_utils.py:16-58`）：用 `next_observations[i] != observations[i+1]` 判定 episode 边界；`masks = 1 - dataset['terminals']`；**antmaze 的 rewards 统一减 1**（`:37`）；最后一条 `terminals=1`。
- 在线阶段 D4RL antmaze 的 reward 再减 1（`main.py:149-153`），与离线平移保持一致。

### 3.5 `utils/evaluation.py`

- `supply_rng`（`:8-16`）每次调用 split 一个持续 RNG；`evaluate`（`:37-105`）默认 `eval_temperature=0`（`:44`），在 `:74` 传给 `agent.sample_actions`，动作再 clip 到 `[-1,1]`（`:76`）。
- 统计只在**每个 episode 结束时**从 `info` 取（`:96-100`），因此 `evaluation/episode.return`、`evaluation/episode.length`、`evaluation/episode.normalized_return`（D4RL）、OGBench 的 `success` 等最终步 info 指标是报告的主指标。
- `config` 参数实际未使用（`:40`），视频逻辑 `:82-84` 只在 `num_video_episodes>0` 时触发。

### 3.6 `utils/log_utils.py`

- `setup_wandb`（`:62-93`）：`mode='online'` 默认（`:68`），`save_code=True`，`settings=wandb.Settings(start_method='thread', _disable_stats=False)`；`main.py:61` 固定 project=`FLAGS.project_name or 'FINO'`、group=env_name、job_type=agent 名。
- `CsvLogger`（`:12-37`）在 `save_dir/train.csv`、`eval.csv` 写指标，表头由第一次 log 的行决定；`wandb.Image/Video/Histogram` 类型被过滤掉（`:19,26-28`）。
- 视频辅助 `get_wandb_video`（`:118-148`）在 `main.py:216-218` 被注释掉，`video_episodes` 默认 0，因此默认不渲染。

### 3.7 检查点

- 保存目录：`{save_dir}/{AGENT}/{env_name}/{seed}/1000000/<module>`；恢复时 `--restore_path=... --restore_epoch=1000000`（`main.py:30-31,63,113-114`）。
- **默认只有 `i % 1e6 == 0` 时保存**（`main.py:230`），默认总步数 1.5e6 → 实际只在 `i=1e6` 保存一次（正好是离线阶段结束）；想保留在线阶段模型必须显式改 `save_interval`。
- 只保存 network params；`config`（在 flags.json 中另存）、β、optimizer step 不保存（`utils/flax_utils.py:171-188`）。

### 3.8 FINO 配置项全表（`fino.py:338-360`）

| 配置 | 默认值 | 说明 |
|---|---|---|
| `agent_name` | `'fino'` | 注册表键 |
| `lr` | 3e-4 | 所有网络共用 Adam |
| `batch_size` | 256 | |
| `actor_hidden_dims` / `value_hidden_dims` | (512,512,512,512) | |
| `layer_norm` | True | critic 用 LayerNorm |
| `actor_layer_norm` | False | flow 网络不用 LayerNorm |
| `discount` | 0.99 | README 中 humanoidmaze/antsoccer/antmaze-giant 改 0.995 |
| `tau` | 0.005 | target critic 软更新 |
| `q_agg` | `'mean'` | 可选 `'min'`（README antmaze 系列） |
| `alpha` | 10.0 | 蒸馏项权重（按环境 3~10000） |
| `flow_steps` | 10 | 教师 Euler 步数，也决定蒸馏目标精度 |
| `normalize_q_loss` | False | 打开后 `q_loss` 用 `1/mean|q|` 归一化 |
| `encoder` | placeholder(None) | 视觉编码器名（`impala*`） |
| `noise_scale` | 0.1 | 即论文 η；t=1 处注入噪声 std |
| `beta` | 10.0（硬编码于 `create`，非 config） | 在线 softmax 温度初值 |
| `ob_dims` / `action_dim` | placeholder | `create()` 回填 |

---

## 4. 与仓库内 baseline 的实现差异

所有 baseline 共享同一套 `main.py` 循环、`Dataset/ReplayBuffer`、`evaluate`、`TrainState`、`ModuleDict`；差异集中在 `agents/*.py`。

### 4.1 网络与损失结构对比

| 维度 | **FINO** | FQL | IFQL | IQL | SAC | ReBRAC |
|---|---|---|---|---|---|---|
| 策略表示 | flow（双策略：教师+一步） | flow（双策略） | flow（单策略，多步） | 高斯（对角） | 高斯（对角/状态相关 std） | 确定性（const std + tanh） |
| 策略网络 | `actor_bc_flow`(带 t) + `actor_onestep_flow` | 同 FINO | `actor_flow`(带 t) | `actor` | `actor` + `LogParam alpha` | `actor` + `target_actor` |
| 价值网络 | 2-Q 集成 | 2-Q 集成 | 1-V + 2-Q | 1-V + 2-Q | 2-Q 集成 | 2-Q 集成 |
| actor 损失 | `L_bc + α·L_distill + L_Q` | 同 FINO（无噪声注入） | `L_bc`（纯 BC flow） | AWR 或 DDPG+BC | `E[α·logπ − Q]` + `α` loss | `−λQ + α_actor·MSE` |
| critic 目标动作 | 一步策略（`sample_batch_actions`） | 一步策略（`sample_actions`） | **V(s')**（IQL 式，无策略采样） | **V(s')**（expectile） | 当前策略采样 + 可选 entropy backup | `target_actor.mode()+clip噪声` + critic BC 惩罚 |
| 目标网络 | 仅 `target_critic` | 仅 `target_critic` | 仅 `target_critic`（用于 V 的 Q 目标） | 仅 `target_critic` | 仅 `target_critic` | `target_critic` + `target_actor` |
| 噪声注入 | **有**（`fino.py:56-58`） | 无 | 无 | 无 | 无 | 无（`actor_noise` 是 target 平滑噪声，不是 loss 注入） |
| 熵/温度 | **β 由 GMM 熵自适应**（评估期更新） | 无 | 无 | 无 | 可学习 `alpha`（`sac.py:64-67`） | 无 |
| 关键 config | `alpha, flow_steps, noise_scale, q_agg` | `alpha, flow_steps, q_agg` | `expectile, num_samples=32, flow_steps` | `expectile, actor_loss, alpha, const_std` | `target_entropy_multiplier, tanh_squash, backup_entropy, q_agg` | `alpha_actor, alpha_critic, actor_freq, actor_noise` |

### 4.2 动作采样 / 在线行为对比

| 维度 | FINO | FQL | IFQL | IQL | SAC | ReBRAC |
|---|---|---|---|---|---|---|
| `sample_actions` | K 个噪声 → 一步策略 → 按 `softmax(βQ)`（在线）或 `argmax(Q)`（eval） | 单噪声 → 一步策略（无 Q 选择） | 32 个噪声 → 完整 flow → `argmax` min-Q | 高斯采样（temperature 缩放 std） | 高斯采样（temperature 缩放 std） | 确定性 mode + 截断噪声 × temperature |
| 每步推理成本 | K 次一步网络 + K 次 Q | 1 次一步网络 | 32×`flow_steps` 次网络 | 1 次 | 1 次 | 1 次 |
| 在线探索来源 | 候选动作 + softmax 温度 + 噪声注入的宽分布 | 一步策略的随机噪声 | 多步 flow 随机性 | 高斯 std | 高斯 std + 可学习温度 | 注入动作噪声 |
| 在线特殊逻辑 | `update_entropy`/β（`main.py:220-224`） | 无 | 无 | 无 | 无 | `actor_freq=2` 隔步更新 actor，target 同步（`main.py:132,182`；`rebrac.py:100-135`） |

### 4.3 与 FQL 的逐点 diff（FINO 相对 FQL 的全部改动）

1. **loss 输入加噪**：`fino.py:56-58` 新增 `normal_noise/noise_scale/normal_samples`，FQL 直接把 `x_t` 送入网络（`fql.py:58`）。
2. **新增 `noise_scale` 配置**（`fino.py:357`；FQL 无）。
3. **critic 目标动作抽成 `sample_batch_actions`**（`fino.py:134-151`），FQL 直接复用 `sample_actions`（`fql.py:25`）；两者在 FQL 中行为等价（FQL 的 `sample_actions` 没有 Q 选择），FINO 这样写是为了区分“TD 用的单候选动作”与“交互用的多候选 Q 采样”。
4. **`sample_actions` 重写为 Q-guided 多候选 + 温度分支**（`fino.py:153-197` vs `fql.py:136-153`），并新增 `q_agg` 在候选选择上的使用。
5. **新增 `beta` 字段、`update_entropy`、`estimate_entropy_sklearn`**（`fino.py:21,215-232,310-335`），并在 `main.py:220-224` 接线；FQL 完全没有在线熵适配。
6. **`create` 末尾 `beta=10.0`**（`fino.py:307`）。

其余（critic 损失、蒸馏、Q 项、网络初始化、target 更新）与 FQL 逐行一致，可作为“最小对照实验”的基线：**FINO = FQL + 噪声注入 + Q-guided/熵采样**，复现报告中可以分别做两个消融（η=0；固定 β / 无 Q 选择）。

### 4.4 其他 baseline 的实现要点（用于对比表脚注）

- **IFQL**（`ifql.py`）：actor 只做 BC flow（`ifql.py:60-76`），critic target 用 `value(s')`（IQL 式 expectile，`ifql.py:31-58`），`sample_actions` 采 32 个噪声、做 10 步 flow、再取 min-Q 最大者（`ifql.py:123-155`；`num_samples=32` 见 `:245`）。它是“IDQL 的 flow 版”，也是 FINO 噪声注入实验里最直接的“无 Q-guided 一步策略”对照。
- **IQL**（`iql.py`）：`value_loss` 用 expectile（`:28-40`），`critic_loss` 用 `next_v`（`:42-55`）；actor 支持 `awr`（`:59-82`，`exp(α·adv)` 截断到 100）与 `ddpgbc`（`:83-110`，`−λQ − α·logπ`），默认 `actor_loss='awr'`（`:255`）。
- **SAC**（`sac.py`）：可学习温度 `LogParam`（`:181`），`target_entropy = -0.5·d`（`:161-162`），`actor_loss = E[α·logπ − Q]` + `α` loss（`:53-84`），`backup_entropy` 默认 False（`:221`）。
- **ReBRAC**（`rebrac.py`）：critic 目标加 `α_critic·||a'−a'_data||²` 惩罚（`:41-42`）；actor 损失 `−λQ + α_actor·MSE`（`:66-73`）；`actor_freq=2` 时奇数步只更新 critic（`:100-110`、`:121-135`，`main.py:132,182` 负责传 `full_update`）；`alpha_actor/alpha_critic` 默认 0.0（`:236-237`），必须命令行覆盖才是真正的 ReBRAC。

---

## 5. 复现要点与潜在坑

> 本章按“先决条件 → 启动 → 运行期行为 → 代码级坑”排列；标 **【已核实】** 的来自静态代码，标 **【待运行验证】** 的需要在 Lead 的环境里确认。

### 5.1 环境与依赖

1. **`requirements.txt` 是 Linux 风格的，Windows 上不能全量安装**：`jax[cuda12]`（第 3 行）需要 CUDA；`mujoco-py==2.1.2.14`（第 10 行）不支持 Windows；`d4rl`（第 12 行）依赖 `gym==0.23.1`、`numpy==1.26.4`、`shimmy`（第 13-17 行），在 Windows 上通常装不上/不可用。**建议 Windows 复现只跑 OGBench**（`ogbench==1.1.0` + `mujoco==3.3.0` + CPU jax），D4RL 放到 WSL/Linux。**【已核实：requirements.txt】**
2. **Python 版本**：README 第 5 行写 Python 3.9；`flax>=0.8.4` + `jax>=0.4.26` 对 3.9~3.12 基本可用，但 `.venv` 若是 3.13 需注意 jaxlib 轮子可用性。**【待运行验证】**
3. **OGBench 数据集下载**：`ogbench.make_env_and_datasets`（`envs/env_utils.py:105`）首次运行会下载数据集到 `~/.ogbench/data`，需要网络和磁盘；离线环境需预下载。**【待运行验证】**
4. **`orbax` API**：`orbax.checkpoint.PyTreeCheckpointer`（`utils/flax_utils.py:183,204`）在较新 orbax 中可能变更/废弃；若报错需锁定 orbax 版本。**【待运行验证】**
5. **`distrax`**：`utils/networks.py:4,74,145-147` 依赖 distrax；jax 版本升级后 distrax 偶有兼容问题。**【待运行验证】**

### 5.2 启动与命令（必须从 `FINO/` 目录运行）

- 默认命令：`python main.py`（README 第 16 行），等价于 `--agent=agents/fino.py --env_name=cube-double-play-singletask-v0`（`main.py:28,49`）。
- 指定任务/超参：`python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min`。
- **相对路径依赖**：`--agent` 默认 `agents/fino.py`（`main.py:49`）、`save_agent`/`restore_agent` 用 `os.getcwd()`（`utils/flax_utils.py:179,200`）→ 必须在 `E:\FINO\FINO` 下运行；否则 config 找不到或 checkpoint 存到奇怪位置。**【已核实】**
- **冒烟测试建议**（不需要完整 1.5M 步）：`--offline_steps=2000 --online_steps=1000 --eval_interval=500 --eval_episodes=3 --log_interval=100 --save_interval=1000 --video_episodes=0`；注意 `eval_interval=0` 会完全关闭评估（`main.py:201`），也就不会触发 β 更新（`main.py:220` 在评估块内）。
- **默认预算很大**：`offline_steps=1e6`、`online_steps=5e5`（`main.py:33-34`），每步一次 jit update + 每 50k 步 50 个 episode 评估；CPU 上不可行，需 GPU 或缩减步数/评估数，报告里要写清实际预算。**【待运行验证】**
- **保存/恢复**：默认 `save_interval=1e6`、`restore_epoch=1e6`（`main.py:31,38`）；默认总步数下只在 `i=1e6` 存一次。若要“先离线训练再在线微调”的两段式实验，应显式提高保存频率或延长 `save_interval` 的倍数关系（`i % save_interval == 0` 只在整数倍触发）。**【已核实】**

### 5.3 wandb 在线模式（最容易卡住的第一关）

- `utils/log_utils.py:68` 把 `mode='online'` 写死并显式传给 `wandb.init`；`main.py:61` 无条件调用。**显式参数会覆盖 `WANDB_MODE` 环境变量**，所以仅设置 `WANDB_MODE=offline` 不一定生效（取决于 wandb 版本对显式参数的优先级，建议不要依赖）。**【已核实（代码）；行为待验证】**
- 处理方式（三选一）：
  1. 先 `wandb login` 并保证能访问外网；
  2. 把 `utils/log_utils.py:setup_wandb` 的 `mode` 改/传为 `'offline'`（或加一个 env 判断）；
  3. 临时把 `main.py:61` 的 `setup_wandb` 调用替换为不初始化 wandb，并给 `wandb.log` 打桩（不推荐，日志会丢）。
- 另外 `save_code=True` 会尝试打包代码上传，离线模式下只是本地缓存，不影响训练。
- 若完全不想要 wandb，`main.py` 里 `wandb.log` 出现 3 处（`:197,226`），需要同时处理。

### 5.4 `MUJOCO_GL=egl` 与 Windows

- `main.py:22` 在 import 后立刻 `os.environ["MUJOCO_GL"] = "egl"`（覆盖用户环境变量），EGL 是 Linux/NVIDIA 的渲染后端；Windows 上应使用 `glfw` 或 `osmesa`。**【已核实（代码）；是否必然报错取决于是否触发渲染】**
- 本次复现默认 `video_episodes=0`（`main.py:41`）、`evaluate` 不渲染（`utils/evaluation.py:82-84`），OGBench 环境创建本身通常不需要 GL context，因此**可能不报错**；但一旦开视频/渲染就会失败。若报 `Cannot initialize EGL`/`GLFW` 相关错误，处理方式：把 `main.py:22` 改成 `os.environ["MUJOCO_GL"] = "glfw"`（Windows）或 `"osmesa"`（无头），或改用 WSL/Linux。**【待运行验证】**
- 该行是**官方代码硬编码**，复现报告应注明“为 Windows 做了这一行的本地修改”，并保证其余代码不动。

### 5.5 `eval_temperature=0` 分支

- `evaluate` 默认 `eval_temperature=0`（`utils/evaluation.py:44`，`main.py:204-211` 不传该参数，用默认值），传给 `sample_actions` 后：
  - **FINO**：`lax.cond(temperature==0)` 走 `to_eval` → `actions[jnp.argmax(q)]`（`fino.py:188-194`），即**在 K 个候选里取 Q 最大的**，动作确定；
  - **FQL**：单噪声一步策略，动作由噪声决定（仍随机，因为 `sample_actions` 每次 split 新 key）；
  - **IQL/SAC**：`Actor.__call__(temperature=0)` → `scale_diag=0`（`utils/networks.py:145`）→ `dist.sample()` 返回均值（确定性）；`tanh_squash` 时同理；
  - **ReBRAC**：`mode() + noise*temperature`，温度 0 → 纯 mode。
- 因此**评估指标不是“随机策略的期望”而是近似确定性策略**；报告对比 baseline 时要意识到这一点（尤其 FQL 在 eval 时仍随机、FINO 是 argmax 候选，两者不完全同口径）。**【已核实】**
- 潜在 bug：`to_eval` 的 `jnp.argmax(q)` 没有指定 `axis`，`q` 形状为 `(K, B)`，**仅当 B=1 时正确**；`evaluate` 每次只传单个 observation（`fino.py:160-161` 会补 batch 维），所以评估路径没问题，但若复现脚本传入批量状态会取到错误动作（JAX 越界 clamp，不报错）。**【已核实（代码逻辑）】**
- `main.py:143` 在线交互用 `temperature=1`，与评估不同，注意区分。

### 5.6 D4RL 相关

- **不能与 OGBench 混装**：`env_utils.py:119-134` 会 `import d4rl`/`d4rl.hand_manipulation_suite`；Windows 上大概率不可用。**【已核实】**
- **reward 双重平移要分清阶段**：D4RL antmaze 离线数据 `rewards -= 1`（`d4rl_utils.py:37`）；在线交互时 `main.py:149-153` 对 `antmaze + (diverse|play|umaze)` 再减 1。两者是**同一约定**（不是 bug），但改环境名/改数据集时要保持一致，否则 reward 尺度会错。**【已核实】**
- D4RL 环境构造：`gymnasium.make('GymV21Environment-v0', env_id=...)`（`d4rl_utils.py:11`），依赖 `shimmy[gym-v21]` 与 `gym==0.23.1`；`gymnasium>=1.0` 会破坏 d4rl（requirements 第 13 行注释也这么说）。**【已核实】**
- D4RL 没有 `val_dataset`（`env_utils.py:118,127,134`），因此 `main.py:189-192` 的验证损失日志不会出现；报告里不要期待 `validation/*`。**【已核实】**

### 5.7 代码级坑（静态阅读发现，建议复现报告单列一节）

1. **β 不入 checkpoint**（§2.7.2）：`save_agent` 只存 params（`utils/flax_utils.py:171-188`），restore 后 β=10（`fino.py:307`）。若做“离线 ckpt → 在线微调”两段式，β 会重置；如需保留，得把 `beta` 也存下来（或在线阶段不 reset）。**【已核实】**
2. **β 更新频率低且随机数固定**（§2.7.2）：只在 `i>offline_steps` 的 eval 时刻触发，默认 50k 步一次（在线 500k → 最多 10 次更新）；`main.py:221` 固定 `PRNGKey(FLAGS.seed)`，每次熵采样用同一组 key。分析 β 曲线时要考虑这两点。**【已核实】**
3. **`update_entropy` 开销**：200 次 `sample_actions` × B=256 × K 候选 + 256 次 GMM 拟合；`one_chunk` 的 `@jax.jit` 定义在函数内部，每次调用重新 trace。评估阶段会明显变慢，建议在报告里记录 eval 耗时或降低 `n_samples`（需改代码，非 config）。**【已核实】**
4. **`save_interval` 默认值导致只存一次**（§5.2）：`i % 1e6 == 0` 在 1.5M 总步数下只在 i=1e6 触发。**【已核实】**
5. **`sample_actions` 的 `to_eval` 批量 bug**（§5.5）。**【已核实】**
6. **`update_entropy` 的 β 更新方向/公式与 ar5iv Eq.(9) 印刷形式不一致**（§2.7.2）：建议报告注明并做消融。**【已核实（代码）；论文原文符号待核对 PDF/附录】**
7. **`main.py:54-56` 的 `get_exp_name` 被覆盖**：wandb run name 只含 `FINO_{seed}`，多个 run 会重名（wandb 自动加后缀），CSV 目录用 `exp/{AGENT}/{env}/{seed}`，**同一 env+seed 重跑会覆盖 `train.csv`/`eval.csv`/checkpoint**。复现实验建议每次清空或改 `save_dir`。**【已核实】**
8. **`fino.py:231` 用 `max(0, ...)` 但无上界**：β 可一直增大；结合 §2.7.2 的固定 key，β 轨迹可能受初值和更新次数影响明显。**【已核实】**
9. **`q_agg` 只作用于 target 和候选选择，不作用于 actor 的 Q 项**：actor 的 `q = mean(qs)`（`fino.py:71`），因此 `--agent.q_agg=min` 不会让 actor 变保守，只影响 TD 目标与动作选择。做消融时别误读。**【已核实】**
10. **`normalize_q_loss` 默认 False**：Q 项与 reward 尺度直接相关；D4RL antmaze 的 `-1` reward 平移会改变 Q 尺度，α 的敏感度可能因此变化。若某任务训练不稳定，可试 `--agent.normalize_q_loss=True`。**【已核实】**
11. **`balanced_sampling` 默认 0**：在线数据直接混入离线 buffer；若做“在线数据比例”消融，需显式 `--balanced_sampling=1`（`main.py:83-92,173-179`）。**【已核实】**
12. **frame_stack/p_aug 仅对视觉有意义**：`main.py:44-45` 默认 None，非视觉 OGBench/D4RL 保持 None 即可。**【已核实】**
13. **`evaluate` 统计口径**：只取 episode 最后一步 info（`evaluation.py:96-100`），所以 `evaluation/episode.return` 是 return；OGBench 用 `success` 指标时，报告要确认取的是 `evaluation/success` 还是 `evaluation/episode.*`（`flatten(info)` 的键取决于 ogbench 版本）。**【待运行验证】**
14. **`main.py:22` 在 import 后立即设置 `MUJOCO_GL`**（§5.4）。**【已核实】**
15. **在线 reward 平移的匹配条件较宽**：`main.py:149-150` 的条件是 `'antmaze' in env_name and ('diverse' or 'play' or 'umaze' in env_name)`，因此任何名字里含 `umaze` 的 antmaze 环境都会被减 1。README 的 OGBench 命令只用了 large/giant（`antmaze-large-navigate`、`antmaze-giant-navigate`），不满足该条件；但如果复现时跑 OGBench 里其他含 `umaze` 的任务（若存在），会与离线数据的 reward 定义不一致，需在报告中记录 reward 约定。**【已核实（条件）；OGBench 命名待验证】**
16. **验证集损失只用于日志**：`main.py:189-192` 调 `agent.total_loss(val_batch, grad_params=None)`，不产生梯度、不更新参数；且只有 OGBench 有 `val_dataset`。**【已核实】**

### 5.8 性能/预算建议（写给 Lead 的复现计划）

- **最小可跑通**：OGBench 一个 `cube-double-play-singletask-v0`（或 `antmaze-large-navigate-singletask-v0`）离线 1e5 步 + 在线 1e4 步，`eval_interval=1e4`、`eval_episodes=10`，先确认全链路（数据加载 → 训练 → 评估 → β 更新 → 保存）。
- **论文量级复现**：1e6 离线 + 5e5 在线、50 eval episodes、每 50k 评估；单 seed 在单卡 GPU 上仍需数小时到一天量级（含 10 步 Euler 蒸馏目标与在线 GMM 熵评估），多 seed（论文 10 seeds）需要并行。**【待运行验证】**
- **关键对照**：`--agent=agents/fql.py`（同 config 但无噪声注入与 β）是 FINO 的天然对照；再跑 `--agent.noise_scale=0`（FINO 但 η=0）验证噪声注入的独立贡献；`--agent.alpha` 按 §6 表逐任务设置。

---

## 6. 论文/README 实验设置与超参整理

> 来源：`README.md:20-113`；代码默认值来自各 `get_config()`。README 命令**没有覆盖** `offline_steps/online_steps/batch_size/lr` 等，因此这些取 `main.py` 与 `fino.py` 的默认值。

### 6.1 OGBench（默认任务别名，README 第 20-36 行）

| 环境（task 后缀省略时用默认 task） | alpha | q_agg | discount | README 行 |
|---|---:|---|---:|---|
| `humanoidmaze-medium-navigate-singletask` | 100 | mean（默认） | 0.995 | 23 |
| `humanoidmaze-large-navigate-singletask` | 30 | mean | 0.995 | 25 |
| `antmaze-large-navigate-singletask` | 10 | **min** | 0.99（默认） | 27 |
| `antmaze-giant-navigate-singletask` | 10 | **min** | 0.995 | 29 |
| `antsoccer-arena-navigate-singletask` | 10 | mean | 0.995 | 31 |
| `cube-double-play-singletask` | 300 | mean | 0.99 | 33 |
| `puzzle-4x4-play-singletask` | 1000 | mean | 0.99 | 35 |

> **注意 1**：README 第 38-39 行的 NOTE 更正了论文中 `antsoccer-arena` 的 α：论文写 30，实际应使用 **10**。
> **注意 2**：全任务列表（task1~task5）见 `README.md:45-88`，α/q_agg/discount 与上表逐任务一致；`humanoidmaze` 用了 `discount=0.995`，`antmaze-giant` 也是，其余 OGBench 任务用默认 0.99。
> **注意 3**：README 中 `cube-double-play-singletask-v0` 对应默认 task2、`puzzle-4x4-play` 对应默认 task4、`antsoccer-arena` 对应默认 task4（见 README 注释）。

### 6.2 D4RL（README 第 90-112 行）

| 环境 | alpha | q_agg | discount | README 行 |
|---|---:|---|---:|---|
| `antmaze-umaze-v2` | 10 | mean | 0.99 | 93 |
| `antmaze-umaze-diverse-v2` | 10 | mean | 0.99 | 95 |
| `antmaze-medium-play-v2` | 10 | mean | 0.99 | 97 |
| `antmaze-medium-diverse-v2` | 10 | mean | 0.99 | 99 |
| `antmaze-large-play-v2` | 3 | mean | 0.99 | 101 |
| `antmaze-large-diverse-v2` | 3 | mean | 0.99 | 103 |
| `pen-cloned-v1` | 1000 | **min** | 0.99 | 105 |
| `door-cloned-v1` | 1000 | **min** | 0.99 | 107 |
| `hammer-cloned-v1` | 1000 | **min** | 0.99 | 109 |
| `relocate-cloned-v1` | 10000 | mean（README 未写 q_agg） | 0.99 | 111 |

> D4RL 的 `eval` 指标依赖 `EpisodeMonitor` 的 `normalized_return`（`env_utils.py:48-51`），报告里用 `evaluation/episode.normalized_return`。

### 6.3 全局默认超参（`main.py` + `fino.py` + `fql.py` 默认值）

| 类别 | 参数 | 值 |
|---|---|---|
| 训练长度 | `offline_steps` / `online_steps` | 1,000,000 / 500,000（`main.py:33-34`） |
| buffer | `buffer_size` | 1,500,000（`main.py:35`） |
| 日志/评估 | `log_interval` / `eval_interval` / `save_interval` | 5,000 / 50,000 / 1,000,000（`main.py:36-38`） |
| 评估 | `eval_episodes` / `video_episodes` | 50 / 0（`main.py:40-41`） |
| 优化 | `lr` / `batch_size` | 3e-4 / 256 |
| 网络 | actor/value hidden | (512,512,512,512) |
| 网络 | `layer_norm` / `actor_layer_norm` | True / False |
| 折扣 | `discount` / `tau` | 0.99 / 0.005 |
| Q 聚合 | `q_agg` | mean（antmaze 系列 min） |
| FINO 蒸馏 | `alpha` | 10（按任务 3~10000） |
| flow | `flow_steps` | 10 |
| 噪声注入 | `noise_scale`（η） | 0.1 |
| Q loss | `normalize_q_loss` | False |
| 熵采样 | β 初值 | 10.0（`fino.py:307`） |
| 熵采样 | 候选数 K | `min(10, (A+1)//2)`（`fino.py:163`） |
| 熵估计 | `n_samples` / `n_components` / 更新步长 | 200 / 3 / 0.1（`fino.py:215,231`） |
| 熵估计 | 目标熵 H̄ | `-action_dim`（隐式，`fino.py:231`） |

### 6.4 论文与 README 的超参对应关系（用于复现报告）

- `α` 在论文 Eq.(5) 中是蒸馏权重，README 按环境给 3~10000；**α 越大越贴教师（保守）**，越小的任务（antmaze）通常越需要探索。
- `q_agg=min` 在 antmaze/adroit-cloned 上使用，属于“双重保守”配置（target 用 min、候选选择也用 min）；消融时注意它同时影响两个位置（§5.7 第 9 条）。
- `discount=0.995` 只用于 humanoidmaze 与 antmaze-giant（长时程任务），其余 0.99。
- `η=0.1` 全局固定（论文 §4.3）；`N_sample≈d/2` 全局固定。
- **README 未给出 online_steps 的逐任务改写**：命令均使用默认 500k；论文附录 G 若给出不同在线预算，需以论文/附录为准（本次未能获取到附录全文，建议 Lead 核对 PDF 附录 G 后在报告中注明）。**【待核对】**

---

## 7. 给复现报告的建议素材（可直接改写）

1. **方法章节**：用 §2.2-2.8 的公式—代码对照，重点写“Eq.(7) 噪声注入 + Eq.(5) 蒸馏与 Q 最大化”的两级结构，并明确“噪声注入发生在数据端 t→1”这一与常见直觉相反的细节（§2.3.2 订正）。
2. **实现章节**：用 §2.9 的单步伪代码 + §1.2 的 main.py 流程 + §2.10 的公式映射表。
3. **实验设置章节**：用 §6 的三张表 + §5.2 的实际预算（写清实际跑了多少 offline/online steps、eval episodes、seeds）。
4. **问题与修正章节**：用 §5.4（Windows/MUJOCO_GL）、§5.3（wandb offline）、§5.5（eval temperature=0 口径）、§5.7（β 不保存、save_interval、固定 key 等）。
5. **消融建议**（与仓库代码可直接对应）：
   - 无噪声注入：`--agent.noise_scale=0`（等价 FQL，但保留 Q-guided/β）与 `--agent=agents/fql.py`（完整 FQL）两条对照；
   - 无 Q-guided：临时把 `fino.py:181-186` 换成单候选（或用 `temperature=0` 采样做在线，仅作为分析）；
   - 无熵自适应：固定 β（把 `main.py:220-224` 注释掉或 β 不更新）；
   - 理论 schedule vs 指数 schedule：把 `fino.py:57` 换成 `sqrt(max(0,(η²−2η)t²+2ηt))`（论文 Eq.7 的理论方差表），验证性能差异——这是本仓库与论文公式之间**最值得报告的一处实现偏差**。
6. **指标**：OGBench 用 `evaluation/success`（或最终 `evaluation/episode.*`，以实际 run 的 csv 为准）；D4RL 用 `evaluation/episode.normalized_return`；同时记录 `actor/bc_flow_loss`、`actor/distill_loss`、`actor/q_loss`、`actor/q`、`critic/critic_loss`、`evaluation/entropy`、`evaluation/beta`（`fino.py:84-91` 与 `main.py:220-224` 已记录）。

---

## 附录 A：代码引用索引（按主题）

| 主题 | 位置 |
|---|---|
| FINO 损失（BC+注入+蒸馏+Q） | `agents/fino.py:46-91` |
| 噪声注入 schedule | `agents/fino.py:56-58`；`agents/fino.py:357` |
| 蒸馏教师积分（Euler） | `agents/fino.py:199-213` |
| critic TD + target | `agents/fino.py:23-44,112-132` |
| 一步策略训练用动作 | `agents/fino.py:134-151` |
| Q-guided 多候选采样 | `agents/fino.py:153-197` |
| 熵估计 + β 更新 | `agents/fino.py:215-232,310-335` |
| agent 构造/网络定义 | `agents/fino.py:234-307` |
| 超参默认值 | `agents/fino.py:338-360` |
| 训练主循环/在线/评估/保存 | `main.py:52-234` |
| 配置系统 | `main.py:24-49`；`utils/log_utils.py:53-59` |
| 网络模块 | `utils/networks.py:33-234` |
| 优化器/训练状态/ckpt | `utils/flax_utils.py:18-210` |
| 数据集/ReplayBuffer | `utils/datasets.py:34-173` |
| 环境分派/包装 | `envs/env_utils.py:13-155` |
| D4RL 数据转换 | `envs/d4rl_utils.py:9-58` |
| 评估 | `utils/evaluation.py:8-105` |
| 日志 | `utils/log_utils.py:12-148` |
| baseline：FQL / IFQL / IQL / SAC / ReBRAC | `agents/fql.py` / `ifql.py` / `iql.py` / `sac.py` / `rebrac.py` |

## 附录 B：本次分析未覆盖/待验证项

1. 论文附录（尤其 Appendix D 噪声注入分析、E.1 熵估计细节、G 超参与逐任务 online 预算）本次只获取到 arXiv HTML 正文（页面在实验表格处被截断），附录数值未核对；建议 Lead 以 PDF 为准补全 §6.4。
2. 所有运行期结论（依赖是否装好、OGBench 数据集是否可下载、Windows 上 `MUJOCO_GL=egl` 是否真的报错、eval 的 `success` 键名、实际吞吐）均标为【待运行验证】，本次未运行任何训练。
3. `README.md` 与 `requirements.txt` 的依赖组合是否能在 Windows 上安装成功，取决于 Lead 的实际 pip 环境；本文只做了静态推断。
4. 论文 Eq.(9) 的符号/目标熵约定与代码 `fino.py:231` 的差异（§2.7.2）需要作者附录或联系作者确认；复现报告中建议以“代码行为 + 消融”呈现，而不是断言谁对谁错。
