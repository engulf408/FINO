# 补充消融结果（Round 2，2026-10-26）

| group | variant | final | peak | last5 | evals | minutes | beta_last |
|---|---|---|---|---|---|---|---|
| noise_schedule | official exp (eta=0.1) | 1.0 | 1.0 | 0.976 | 31 | 150.3 | nan |
| noise_schedule | no noise (eta=0) | 0.96 | 0.98 | 0.876 | 16 | 127.7 | 11.25 |
| noise_schedule | exp eta=0.3 | 0.96 | 0.96 | 0.748 | 16 | 112.8 | 11.55 |
| noise_schedule | Eq.(7) unscaled | 0.04 | 0.62 | 0.012 | 16 | 136.1 | 13.16 |
| noise_schedule | Eq.(7) peak-matched | 0.94 | 0.94 | 0.756 | 16 | 154.0 | 11.85 |
| noise_schedule | Eq.(7) variance-read | 1.0 | 1.0 | 0.856 | 16 | 158.8 | 11.45 |
| seeds | official seed0 | 1.0 | 1.0 | 0.976 | 31 | 150.3 | nan |
| seeds | official seed1 | 0.96 | 1.0 | 0.912 | 16 | 156.5 | 11.29 |
| seeds | Eq.(7) seed0 | 0.04 | 0.62 | 0.012 | 16 | 136.1 | 13.16 |
| seeds | Eq.(7) seed1 | 0.12 | 0.52 | 0.028 | 16 | 141.4 | 13.14 |
| cross_task | humanoid official | 1.0 | 1.0 | 0.96 | 16 | 127.4 | nan |
| cross_task | humanoid Eq.(7) | 0.0 | 0.0 | 0.0 | 16 | 139.3 | 19.21 |
| qguided_K | K=1 | 0.9 | 0.96 | 0.804 | 11 | 86.0 | 12.89 |
| qguided_K | K=4 (official) | 0.94 | 1.0 | 0.96 | 11 | 70.0 | 12.91 |
| qguided_K | K=10 | 0.96 | 1.0 | 0.968 | 11 | 61.9 | 13.05 |
