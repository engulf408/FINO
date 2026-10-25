"""Publish the staged reproduction repo through the GitHub Git Data API.

The git smart-HTTP endpoint (github.com) is unreachable/flaky from this network, but
api.github.com works, so we build the commit server-side:
  blobs (one per file) -> tree -> commit -> ref refs/heads/main.

Usage:
  python publish_api.py --token-file E:\\FINO\\_gh\\token.txt
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
CTX = ssl.create_default_context()
UA = 'FINO-reproduction-publisher'
RETRIES = 5


def request(method, path, token, payload=None, raw=False, timeout=90):
    url = API + path
    last = None
    for attempt in range(RETRIES):
        try:
            data = None
            headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                       'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': UA}
            if payload is not None:
                data = json.dumps(payload).encode()
                headers['Content-Type'] = 'application/json'
            if raw:
                headers['Accept'] = 'application/vnd.github.raw+json'
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
                body = r.read()
                if raw:
                    return r.status, body
                return r.status, json.loads(body.decode() or '{}')
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors='replace')
            if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                last = f'HTTP {e.code}: {body[:200]}'
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
    ap.add_argument('--token-file', required=True)
    ap.add_argument('--repo', default='FINO-reproduction')
    ap.add_argument('--message', default='FINO reproduction: official code + 4 platform patches, technical analysis, '
                                         'reduced-budget experiments (OGBench antmaze-large FINO/FQL), '
                                         '2D mechanism validation, reports and logs')
    args = ap.parse_args()

    token = open(args.token_file, encoding='utf-8').read().strip()
    if not token:
        sys.exit('empty token file')

    status, user = request('GET', '/user', token)
    if status != 200:
        sys.exit(f'auth failed: {status} {user}')
    login = user['login']
    full = f'{login}/{args.repo}'
    print('authenticated as', login, '| repo', full)

    status, repo = request('GET', f'/repos/{full}', token)
    if status == 404:
        status, repo = request('POST', '/user/repos', token, {
            'name': args.repo, 'private': False, 'auto_init': False,
            'description': 'Reproduction of FINO (ICLR 2026) on Windows + CPU: code patches, analysis, '
                           'reduced-budget experiments, 2D mechanism validation and reports.',
        })
        if status not in (200, 201):
            sys.exit(f'repo creation failed: {status} {repo}')
        print('created repo', repo['html_url'])
    elif status != 200:
        sys.exit(f'repo lookup failed: {status} {repo}')
    else:
        print('repo exists:', repo['html_url'])

    # Git Data API refuses to work on a completely empty repository (409 "Git Repository is empty"),
    # so create a bootstrap commit through the Contents API first, then build the real commit on top.
    status, branches = request('GET', f'/repos/{full}/branches', token)
    if status == 200 and len(branches) == 0:
        print('empty repo -> creating bootstrap commit')
        status, init = request('PUT', f'/repos/{full}/contents/README.md', token, {
            'message': 'Bootstrap repository (content is replaced by the next commit)',
            'content': base64.b64encode(b'# FINO reproduction\n\nContent is being published.\n').decode('ascii'),
        })
        if status not in (200, 201):
            sys.exit(f'bootstrap failed: {status} {init}')
        time.sleep(2)
        status, repo = request('GET', f'/repos/{full}', token)

    branch = repo.get('default_branch') or 'main'
    status, ref = request('GET', f'/repos/{full}/git/ref/heads/{branch}', token)
    if status != 200:
        sys.exit(f'branch {branch} not found: {status} {ref}')
    base_sha = ref['object']['sha']
    print(f'branch {branch} base commit: {base_sha}')

    # 1) find worktree files (.git excluded)
    files = []
    for root, dirs, names in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, REPO_DIR).replace('\\', '/')
            files.append((rel, p))
    files.sort()
    print(f'uploading {len(files)} files ...')

    # 2) blobs
    tree_entries = []
    total = 0
    for i, (rel, path) in enumerate(files, 1):
        content = open(path, 'rb').read()
        total += len(content)
        status, blob = request('POST', f'/repos/{full}/git/blobs', token, {
            'content': base64.b64encode(content).decode('ascii'),
            'encoding': 'base64',
        })
        if status not in (200, 201):
            sys.exit(f'blob failed for {rel}: {status} {blob}')
        tree_entries.append({'path': rel, 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
        print(f'  [{i}/{len(files)}] {rel} ({len(content)} B)')
    print(f'blobs done, {total/1e6:.2f} MB total')

    # 3) tree (paths with slashes are accepted; GitHub materialises nested trees)
    status, tree = request('POST', f'/repos/{full}/git/trees', token, {'tree': tree_entries})
    if status not in (200, 201):
        sys.exit(f'tree failed: {status} {tree}')
    print('tree:', tree['sha'])

    # 4) commit
    status, commit = request('POST', f'/repos/{full}/git/commits', token, {
        'message': args.message,
        'tree': tree['sha'],
        'parents': [base_sha],
        'author': {'name': login, 'email': f'{login}@users.noreply.github.com'},
        'committer': {'name': login, 'email': f'{login}@users.noreply.github.com'},
    })
    if status not in (200, 201):
        sys.exit(f'commit failed: {status} {commit}')
    print('commit:', commit['sha'])

    # 5) update the branch ref to the new commit (fast-forward from the bootstrap commit)
    status, ref = request('PATCH', f'/repos/{full}/git/refs/heads/{branch}', token,
                          {'sha': commit['sha'], 'force': False})
    if status not in (200, 201):
        sys.exit(f'ref update failed: {status} {ref}')
    print(f'branch {branch} ->', commit['sha'])

    # 6) verify
    time.sleep(2)
    status, repo = request('GET', f'/repos/{full}', token)
    print('repo html_url:', repo.get('html_url'), '| default_branch:', repo.get('default_branch'))
    status, tree = request('GET', f'/repos/{full}/git/trees/{commit["sha"]}?recursive=1', token)
    n = len(tree.get('tree', []))
    print('files visible in tree:', n)
    status, readme = request('GET', f'/repos/{full}/contents/README.md', token, raw=True)
    print('README bytes:', len(readme) if status == 200 else f'HTTP {status}')
    if n != len(files):
        print(f'WARNING: expected {len(files)} files, got {n}')


if __name__ == '__main__':
    main()
