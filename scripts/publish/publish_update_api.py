"""Publish the updated reproduction workspace to engulf408/FINO through the GitHub Git Data API.

The git smart-HTTP endpoint is unreachable from this network, so the commit is built server-side:
blobs -> tree -> commit (parent = current remote main) -> update ref.

Usage: python publish_update_api.py --token-file "E:\FINO\_gh\新建 文本文档.txt"
"""
import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

REPO_DIR = r'E:\FINO\_gh\FINO-reproduction'
API = 'https://api.github.com'
FULL = 'engulf408/FINO'
CTX = ssl.create_default_context()
UA = 'FINO-reproduction-updater'
RETRIES = 5


def request(method, path, token, payload=None, raw=False, timeout=90):
    url = API + path
    last = None
    for attempt in range(RETRIES):
        try:
            headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                       'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': UA}
            data = None
            if payload is not None:
                data = json.dumps(payload).encode()
                headers['Content-Type'] = 'application/json'
            if raw:
                headers['Accept'] = 'application/vnd.github.raw+json'
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
                body = r.read()
                return r.status, (body if raw else json.loads(body.decode() or '{}'))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors='replace')
            if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                last = f'HTTP {e.code}: {body[:120]}'
                time.sleep(2 ** attempt + 1)
                continue
            return e.code, {'error': body}
        except Exception as e:  # noqa: BLE001
            last = f'{type(e).__name__}: {e}'
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt + 1)
                continue
            return 0, {'error': last}
    return 0, {'error': last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-file', default=None, help='Path to token file; omit to use GITHUB_TOKEN env')
    ap.add_argument('--message', default='Add full-budget GPU reproduction results (WSL2 + RTX 5070): '
                                         'antmaze-large 1M+500k FINO 100% vs FQL 92%, GPU benchmark/analysis '
                                         'scripts, WSL setup scripts, updated report and PPT')
    args = ap.parse_args()

    if args.token_file:
        token = open(args.token_file, encoding='utf-8').read().strip()
    else:
        token = os.environ.get('GITHUB_TOKEN', '').strip()
    if not token:
        sys.exit('no token: pass --token-file or set GITHUB_TOKEN')
    status, user = request('GET', '/user', token)
    if status != 200:
        sys.exit(f'auth failed: {status} {user}')
    print('authenticated as', user['login'])

    status, repo = request('GET', f'/repos/{FULL}', token)
    if status != 200:
        sys.exit(f'repo not found: {status} {repo}')
    print('repo:', repo['html_url'], '| default branch:', repo['default_branch'])

    status, ref = request('GET', f'/repos/{FULL}/git/ref/heads/{repo["default_branch"]}', token)
    if status != 200:
        sys.exit(f'ref failed: {status} {ref}')
    parent = ref['object']['sha']
    print('parent commit:', parent)

    # Use the local git index so that .gitignore rules (checkpoints, binaries) are respected.
    import subprocess
    tracked = subprocess.run([r'C:\Program Files\Git\cmd\git.exe', '-C', REPO_DIR, 'ls-files', '-z'],
                             capture_output=True, check=True).stdout.decode('utf-8')
    files = []
    for rel in tracked.split('\0'):
        if not rel:
            continue
        p = os.path.join(REPO_DIR, rel.replace('/', os.sep))
        if os.path.exists(p):
            files.append((rel, p))
    files.sort()
    total = sum(os.path.getsize(p) for _, p in files)
    print(f'uploading {len(files)} files ({total/1e6:.2f} MB) ...')

    entries = []
    for i, (rel, path) in enumerate(files, 1):
        content = open(path, 'rb').read()
        status, blob = request('POST', f'/repos/{FULL}/git/blobs', token, {
            'content': base64.b64encode(content).decode('ascii'), 'encoding': 'base64'})
        if status not in (200, 201):
            sys.exit(f'blob failed for {rel}: {status} {blob}')
        entries.append({'path': rel, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
        if i % 20 == 0 or i == len(files):
            print(f'  [{i}/{len(files)}]')

    status, tree = request('POST', f'/repos/{FULL}/git/trees', token, {'tree': entries})
    if status not in (200, 201):
        sys.exit(f'tree failed: {status} {tree}')
    print('tree:', tree['sha'])

    status, commit = request('POST', f'/repos/{FULL}/git/commits', token, {
        'message': args.message, 'tree': tree['sha'], 'parents': [parent],
        'author': {'name': user['login'], 'email': f'{user["login"]}@users.noreply.github.com'},
        'committer': {'name': user['login'], 'email': f'{user["login"]}@users.noreply.github.com'}})
    if status not in (200, 201):
        sys.exit(f'commit failed: {status} {commit}')
    print('new commit:', commit['sha'])

    status, ref2 = request('PATCH', f'/repos/{FULL}/git/refs/heads/{repo["default_branch"]}', token,
                           {'sha': commit['sha'], 'force': False})
    if status not in (200, 201):
        sys.exit(f'ref update failed: {status} {ref2}')
    print('ref updated ->', commit['sha'])

    time.sleep(2)
    status, tree = request('GET', f'/repos/{FULL}/git/trees/{commit["sha"]}?recursive=1', token)
    blobs = [t for t in tree.get('tree', []) if t['type'] == 'blob']
    print('remote blobs:', len(blobs), '| bytes:', sum(t.get('size', 0) for t in blobs))
    status, commits = request('GET', f'/repos/{FULL}/commits?per_page=3', token)
    print('remote commits:', [(c['sha'][:8], c['commit']['message'].splitlines()[0][:60]) for c in commits])


if __name__ == '__main__':
    main()
