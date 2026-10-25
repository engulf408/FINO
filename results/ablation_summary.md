# 论文公式逻辑消融（antmaze-large，GPU 全预算）

| group | variant | label | eval_points | success_final | success_peak | success_last5 | return_final | minutes | beta_last | entropy_last |
|---|---|---|---|---|---|---|---|---|---|---|
| A_noise_schedule | official | 官方指数 schedule（基线） | 31 | 1.0 | 1.0 | 0.976 | -383.4 | 150.3 | nan | nan |
| A_noise_schedule | A2_quad_sched | 论文二次型 Eq.(7) | 16 | 0.04 | 0.62 | 0.012 | -979.4 | 136.1 | 13.16 | -1.51 |
| A_noise_schedule | A3_none_noise | 无噪声 η=0 | 16 | 0.96 | 0.98 | 0.876 | -402.7 | 127.7 | 11.25 | -5.83 |
| A_noise_schedule | A4_eta03 | 指数 schedule η=0.3 | 16 | 0.96 | 0.96 | 0.748 | -398.1 | 112.8 | 11.55 | -4.91 |
| B_beta | B1_resume_official | 官方 β（恢复基线） | 11 | 0.94 | 1.0 | 0.96 | -426.6 | 70.0 | 12.91 | -5.84 |
| B_beta | B2_paper_beta | 论文 Eq.(9) 符号 | 11 | 0.96 | 1.0 | 0.96 | -384.2 | 70.8 | 7.07 | -5.72 |
| B_beta | B3_fixed_beta | 固定 β=10 | 11 | 0.94 | 1.0 | 0.944 | -414.5 | 71.5 | 10.0 | -5.81 |
