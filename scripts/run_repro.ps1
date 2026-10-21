# FINO 本机复现运行脚本（Windows + CPU）
# 用法（执行策略受限，需通过 -Command 调用）：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& 'E:\FINO\scripts\run_repro.ps1' -Run smoke"
# Run 取值：smoke | antmaze-fino | antmaze-fql | restore | all
param(
    [ValidateSet('smoke', 'antmaze-fino', 'antmaze-fql', 'restore', 'all')]
    [string]$Run = 'smoke',
    [int]$Seed = 0
)

$ErrorActionPreference = 'Stop'
$env:FINO_WANDB_MODE = 'offline'
$env:WANDB_SILENT = 'true'
$py = 'E:\FINO\.venv\Scripts\python.exe'
$repo = 'E:\FINO\FINO'
$logdir = 'E:\FINO\logs'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
New-Item -ItemType Directory -Force -Path 'E:\FINO\exp' | Out-Null

function Invoke-Run([string]$tag, [string[]]$Args) {
    $log = Join-Path $logdir ("run_{0}.log" -f $tag)
    "START $(Get-Date -Format o)" | Out-File $log
    Push-Location $repo
    try {
        & $py main.py @Args *>> $log
        "exit=$LASTEXITCODE" | Out-File $log -Append
    } finally {
        Pop-Location
    }
}

switch ($Run) {
    'smoke' {
        Invoke-Run 'pointmaze_fino' @(
            "--env_name=pointmaze-medium-navigate-singletask-v0", "--agent.alpha=10",
            "--seed=$Seed", "--offline_steps=8000", "--online_steps=4000",
            "--eval_interval=1000", "--eval_episodes=10", "--log_interval=500",
            "--save_interval=8000", "--save_dir=E:/FINO/exp"
        )
    }
    'antmaze-fino' {
        Invoke-Run 'antmaze_fino' @(
            "--env_name=antmaze-large-navigate-singletask-v0", "--agent.alpha=10", "--agent.q_agg=min",
            "--seed=$Seed", "--offline_steps=30000", "--online_steps=10000",
            "--eval_interval=2500", "--eval_episodes=10", "--log_interval=1000",
            "--save_interval=30000", "--save_dir=E:/FINO/exp"
        )
    }
    'antmaze-fql' {
        Invoke-Run 'antmaze_fql' @(
            "--agent=agents/fql.py", "--env_name=antmaze-large-navigate-singletask-v0",
            "--agent.alpha=10", "--agent.q_agg=min",
            "--seed=$Seed", "--offline_steps=30000", "--online_steps=10000",
            "--eval_interval=2500", "--eval_episodes=10", "--log_interval=1000",
            "--save_interval=30000", "--save_dir=E:/FINO/exp"
        )
    }
    'restore' {
        Invoke-Run 'pointmaze_restore' @(
            "--env_name=pointmaze-medium-navigate-singletask-v0",
            "--restore_path=E:/FINO/exp/FINO/pointmaze-medium-navigate-singletask-v0/0",
            "--restore_epoch=8000", "--offline_steps=0", "--online_steps=300",
            "--eval_interval=100", "--eval_episodes=2", "--log_interval=100",
            "--save_interval=1000000", "--save_dir=E:/FINO/exp_restore"
        )
    }
    'all' {
        & $PSCommandPath -Run smoke -Seed $Seed
        & $PSCommandPath -Run antmaze-fino -Seed $Seed
        & $PSCommandPath -Run antmaze-fql -Seed $Seed
    }
}

# 训练完成后生成曲线与汇总：
#   E:\FINO\.venv\Scripts\python.exe E:\FINO\scripts\plot_results.py
