#!/bin/bash
# FINO reproduction - WSL Ubuntu base setup, step 1 (v2): HTTP mirror first (minimal rootfs lacks CA certs)
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/5] point apt to Tsinghua over HTTP (bootstrap) ==="
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
  sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' /etc/apt/sources.list.d/ubuntu.sources
  grep -E '^URIs:' /etc/apt/sources.list.d/ubuntu.sources || true
fi

echo "=== [2/5] apt update (HTTP) ==="
apt-get update -qq

echo "=== [3/5] apt install bootstrap tools ==="
apt-get install -y -qq ca-certificates curl python3 python3-venv python3-pip git >/dev/null

echo "=== [4/5] switch apt back to HTTPS after CA certs installed ==="
sed -i 's|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources
apt-get update -qq

echo "=== [5/5] versions & network test ==="
python3 --version
curl -s -o /dev/null -w 'tuna_pip_http=%{http_code} time=%{time_total}s\n' --max-time 20 https://pypi.tuna.tsinghua.edu.cn/simple/jax/
echo "SETUP1_DONE"
