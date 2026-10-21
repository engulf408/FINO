# FINO 复现实验结果汇总（CPU、预算缩减版）

> 论文每任务预算为 1M 离线 + 500k 在线；这里每任务仅 30k + 10k（约 3%）。
> `eval_success_max` 为所有评估点的最高成功率；0 表示缩减预算下策略未学会任务。

| agent | env | seed | offline_steps | online_steps | updates | total_seconds | ms_per_update | bc_flow_loss_first | bc_flow_loss_last | critic_loss_last | q_mean_last | eval_success_max | eval_return_last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FINO | pointmaze-medium-navigate-singletask-v0 | 0 | 8000 | 4000 | 24 | 928.0 | 77.3 | 1.0868 | 1.0221 | 0.2835 | -45.91 | 0.0 | -1000.0 |
| FINO | antmaze-large-navigate-singletask-v0 | 0 | 30000 | 10000 | 40 | 4083.7 | 102.1 | 0.6555 | 0.3539 | 0.1912 | -85.62 | 0.1 | -1000.0 |
| FQL | antmaze-large-navigate-singletask-v0 | 0 | 30000 | 10000 | 40 | 3479.0 | 87.0 | 0.7251 | 0.4046 | 0.2527 | -85.66 | 0.1 | -1000.0 |

