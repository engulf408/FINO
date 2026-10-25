# -*- coding: utf-8 -*-
"""Independent audit of the public repo engulf408/FINO: is it a genuine FINO reproduction?

Uses api.github.com (github.com direct is blocked on this network). Unauthenticated.
Checks: repo metadata, commit history, file tree, key artifacts (CSV/report/PPT/logs),
code patches vs upstream CTID282/FINO, internal consistency of claimed results, and
local-vs-remote hash equality.
"""
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

API = 'https://api.github.com'
H = {'Accept': 'application/vnd.github+json', 'User-Agent': 'fino-repo-audit',
     'X-GitHub-Api-Version': '2022-11-28'}
CACHE = r'E:\FINO\audit_cache'
LOCAL = r'E:\FINO'
os.makedirs(CACHE, exist_ok=True)


def get(path, raw=False, retries=3):
    url = API + path
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=H)
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                return b if raw else json.loads(b.decode() or '{}')
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and i < retries - 1:
                time.sleep(5)
                continue
            return {'_error': f'HTTP {e.code}: {e.read().decode()[:120]}'}
        except Exception as e:  # noqa: BLE001
            if i < retries - 1:
                time.sleep(3)
                continue
            return {'_error': str(e)}


def blob(sha, repo='engulf408/FINO'):
    r = get(f'/repos/{repo}/git/blobs/{sha}')
    if '_error' in r:
        return None
    return base64.b64decode(r.get('content', ''))


def sha256(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    out = {}
    repo = get('/repos/engulf408/FINO')
    print('== repo ==')
    print('full_name:', repo.get('full_name'), '| public:', not repo.get('private'),
          '| created:', repo.get('created_at'), '| pushed:', repo.get('pushed_at'))
    print('description:', (repo.get('description') or '')[:120])
    commits = get('/repos/engulf408/FINO/commits?per_page=10')
    print('\n== commits ==')
    for c in commits:
        print(' ', c['sha'][:8], c['commit']['author']['date'], '|', c['commit']['message'].splitlines()[0][:80])
    tree = get('/repos/engulf408/FINO/git/trees/main?recursive=1')
    blobs = [t for t in tree.get('tree', []) if t['type'] == 'blob']
    paths = {t['path']: t for t in blobs}
    print('\n== tree ==')
    print('files:', len(blobs), '| bytes:', sum(t.get('size', 0) for t in blobs), '| truncated:', tree.get('truncated'))
    groups = {}
    for p in paths:
        groups.setdefault(p.split('/')[0], 0)
        groups[p.split('/')[0]] += 1
    print('top dirs:', dict(sorted(groups.items())))

    # ---------- artifacts ----------
    def read_text(path):
        t = paths.get(path)
        return blob(t['sha']).decode('utf-8', 'replace') if t else None

    print('\n== key artifacts ==')
    for p in ['README.md', 'docs/02_reproduction_report.md', 'docs/01_code_analysis.md',
              'docs/patches.diff', 'results/gpu_summary.md', 'ppt/FINO复现工作与结果分析.pptx',
              'gpu_exp/FINO/antmaze-large-navigate-singletask-v0/0/eval.csv',
              'gpu_exp/FINO/antmaze-large-navigate-singletask-v0/0/train.csv',
              'gpu_exp/FQL/antmaze-large-navigate-singletask-v0/0/eval.csv',
              'gpu_exp/FQL/antmaze-large-navigate-singletask-v0/0/train.csv',
              'logs/gpu_antmaze_fino.log']:
        t = paths.get(p)
        print(f'  {"OK " if t else "MISS"} {p}  {t["size"] if t else "-"} B')

    # CSV consistency
    import pandas as pd
    ev_fino = pd.read_csv(io.BytesIO(blob(paths['gpu_exp/FINO/antmaze-large-navigate-singletask-v0/0/eval.csv']['sha'])))
    tr_fino = pd.read_csv(io.BytesIO(blob(paths['gpu_exp/FINO/antmaze-large-navigate-singletask-v0/0/train.csv']['sha'])))
    ev_fql = pd.read_csv(io.BytesIO(blob(paths['gpu_exp/FQL/antmaze-large-navigate-singletask-v0/0/eval.csv']['sha'])))
    tr_fql = pd.read_csv(io.BytesIO(blob(paths['gpu_exp/FQL/antmaze-large-navigate-singletask-v0/0/train.csv']['sha'])))
    print('\n== result consistency (remote CSVs) ==')
    print(' FINO: eval_points=%d final_success=%.2f peak=%.2f | train_rows=%d max_step=%d total_min=%.1f'
          % (len(ev_fino), ev_fino['evaluation/success'].iloc[-1], ev_fino['evaluation/success'].max(),
             len(tr_fino), tr_fino['step'].max(), tr_fino['time/total_time'].iloc[-1] / 60))
    print(' FQL : eval_points=%d final_success=%.2f peak=%.2f | train_rows=%d max_step=%d total_min=%.1f'
          % (len(ev_fql), ev_fql['evaluation/success'].iloc[-1], ev_fql['evaluation/success'].max(),
             len(tr_fql), tr_fql['step'].max(), tr_fql['time/total_time'].iloc[-1] / 60))
    print(' FINO success series:', [round(x, 2) for x in ev_fino['evaluation/success'].tolist()])
    print(' FQL  success series:', [round(x, 2) for x in ev_fql['evaluation/success'].tolist()])

    # gpu_summary claims vs CSV
    summary = read_text('results/gpu_summary.md') or ''
    print('\n== summary claims vs CSV ==')
    for s in summary.splitlines():
        if 'FINO' in s or 'FQL' in s:
            print(' ', s[:160])

    # ---------- code patches vs upstream ----------
    print('\n== code patches: repo FINO/ vs upstream CTID282/FINO ==')
    up_tree = get('/repos/CTID282/FINO/git/trees/main?recursive=1')
    up_blobs = {t['path']: t for t in up_tree.get('tree', []) if t['type'] == 'blob'} if 'tree' in up_tree else {}
    for p in ['main.py', 'utils/log_utils.py', 'utils/evaluation.py', 'utils/networks.py', 'agents/fino.py']:
        rp = 'FINO/' + p
        repo_text = read_text(rp)
        up_sha = up_blobs.get(p, {}).get('sha')
        up_text = blob(up_sha, 'CTID282/FINO').decode('utf-8', 'replace') if up_sha else None
        if repo_text is None or up_text is None:
            print(f'  {p}: repo={repo_text is not None} upstream={up_text is not None}')
            continue
        print(f'  {p}: identical={repo_text == up_text} | repo_sha={sha256(repo_text.encode())} '
              f'| upstream_sha={sha256(up_text.encode())}')
        if repo_text != up_text:
            for marker in ['FINO-Windows-CPU patch', 'FINO-WSL-PATCH', 'MUJOCO_GL', 'FINO_WANDB_MODE',
                           '2**31 - 1', 'start_method']:
                if marker in repo_text:
                    print(f'      marker present: {marker}')

    # ---------- logs authenticity ----------
    print('\n== log sample ==')
    log = read_text('logs/gpu_antmaze_fino.log') or ''
    lines = log.splitlines()
    print(' lines:', len(lines), '| first:', lines[0][:100] if lines else '')
    for l in lines[1:4]:
        print('  ', l[:120])
    print('  ...')
    for l in lines[-3:]:
        print('  ', l[:120])

    # ---------- local vs remote hash equality ----------
    print('\n== local workspace vs remote (key files) ==')
    checks = ['README.md', 'docs/02_reproduction_report.md', 'docs/patches.diff',
              'results/gpu_summary.md', 'FINO/main.py',
              'gpu_exp/FINO/antmaze-large-navigate-singletask-v0/0/eval.csv',
              'gpu_exp/FQL/antmaze-large-navigate-singletask-v0/0/eval.csv']
    same = 0
    for p in checks:
        lp = os.path.join(LOCAL, p.replace('/', os.sep))
        t = paths.get(p)
        if not os.path.exists(lp) or t is None:
            print('  SKIP', p)
            continue
        local_b = open(lp, 'rb').read()
        remote_b = blob(t['sha'])
        eq = local_b == remote_b
        same += int(eq)
        print(f'  {"MATCH" if eq else "DIFF "} {p}')
    print(f'  => {same}/{len(checks)} matched')

    print('\nAUDIT_DONE')


if __name__ == '__main__':
    main()
