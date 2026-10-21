#!/bin/bash
# FINO reproduction - WSL setup step 4: copy code + datasets, test imports
set -e
echo "=== [1/3] copy repo (code only) -> /opt/fino/FINO ==="
rm -rf /opt/fino/FINO
mkdir -p /opt/fino
cp -r /mnt/e/FINO/FINO /opt/fino/FINO
find /opt/fino/FINO -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
du -sh /opt/fino/FINO

echo "=== [2/3] copy OGBench datasets from Windows cache ==="
mkdir -p ~/.ogbench/data
cp -n /mnt/c/Users/33709/.ogbench/data/*.npz ~/.ogbench/data/ 2>/dev/null || true
ls -lh ~/.ogbench/data | head -6

echo "=== [3/3] import test (expect errors to fix) ==="
cd /opt/fino/FINO
/opt/fino-venv/bin/python - <<'PY'
import traceback
try:
    from agents.fino import FINOAgent
    print('IMPORT_OK fino')
except Exception:
    traceback.print_exc()
try:
    from agents.fql import FQLAgent
    print('IMPORT_OK fql')
except Exception:
    traceback.print_exc()
try:
    from envs.env_utils import make_env_and_datasets
    print('IMPORT_OK env_utils')
except Exception:
    traceback.print_exc()
print('SETUP4_DONE')
PY
