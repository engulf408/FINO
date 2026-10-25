"""Publish the local reproduction repo to GitHub.

Steps:
  1. read a classic PAT from a local token file (never printed, never committed),
  2. resolve the authenticated user via the GitHub API,
  3. create <user>/FINO-reproduction (public) if it does not exist,
  4. amend the commit author to the GitHub login (noreply email),
  5. push main using a temporary credential file that is deleted afterwards.

Usage:
  python publish.py --token-file E:\\FINO\\_gh\\token.txt [--repo FINO-reproduction]
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

GIT = r'C:\Program Files\Git\cmd\git.exe'
REPO_DIR = r'E:\FINO\_gh\FINO-reproduction'
CRED_FILE = r'E:\FINO\_gh\.git-credentials'
API = 'https://api.github.com'


def api(method, path, token, payload=None):
    url = API + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('Accept', 'application/vnd.github+json')
    req.add_header('X-GitHub-Api-Version', '2022-11-28')
    req.add_header('User-Agent', 'FINO-reproduction-publisher')
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')
        return e.code, {'error': body}


def git(*args, env=None):
    cmd = [GIT, '-C', REPO_DIR, *args]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} failed ({p.returncode}):\n{p.stdout}\n{p.stderr}')
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-file', required=True)
    ap.add_argument('--repo', default='FINO-reproduction')
    ap.add_argument('--description',
                    default='Reproduction of FINO (ICLR 2026, Flow Matching with Injected Noise for Offline-to-Online RL) on Windows + CPU: code patches, analysis, reduced-budget experiments, 2D mechanism validation and reports.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--proxy', default='http://127.0.0.1:17890',
                    help='local HTTP proxy (system proxy from Windows Internet Settings)')
    args = ap.parse_args()

    # urllib and git both honour these proxy variables; GitHub is not directly reachable here.
    if args.proxy:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            os.environ[name] = args.proxy
        print('using proxy:', args.proxy)

    token = open(args.token_file, encoding='utf-8').read().strip()
    if not token:
        sys.exit('empty token file')

    status, user = api('GET', '/user', token)
    if status != 200:
        sys.exit(f'token check failed: HTTP {status} {user}')
    login = user['login']
    print(f'authenticated as: {login}')

    full = f'{login}/{args.repo}'
    status, repo = api('GET', f'/repos/{full}', token)
    if status == 404:
        print(f'creating repo {full} (public) ...')
        status, repo = api('POST', '/user/repos', token, {
            'name': args.repo,
            'description': args.description,
            'private': False,
            'has_issues': True,
            'has_wiki': False,
            'auto_init': False,
        })
        if status not in (200, 201):
            sys.exit(f'repo creation failed: HTTP {status} {repo}')
        print('created:', repo['html_url'])
    elif status == 200:
        print('repo already exists:', repo['html_url'])
    else:
        sys.exit(f'repo check failed: HTTP {status} {repo}')

    remote = f'https://github.com/{full}.git'
    try:
        git('remote', 'set-url', 'origin', remote)
    except RuntimeError:
        git('remote', 'add', 'origin', remote)
    print('remote origin ->', remote)

    # author identity: use GitHub login + noreply email
    git('config', 'user.name', login)
    git('config', 'user.email', f'{login}@users.noreply.github.com')
    git('commit', '--amend', '--reset-author', '--no-edit')
    print('commit author set to:', git('log', '-1', '--format=%an <%ae>'))

    if args.dry_run:
        print('dry run: skipping push')
        return

    # temporary credential store (deleted in finally)
    auth = base64.b64encode(f'{login}:{token}'.encode()).decode()
    with open(CRED_FILE, 'w', encoding='utf-8') as f:
        f.write(f'https://x-access-token:{token}@github.com\n')
    try:
        out = git('-c', f'credential.helper=store --file={CRED_FILE}',
                  '-c', f'http.proxy={args.proxy}',
                  'push', '-u', 'origin', 'main')
        print(out)
    finally:
        for p in (CRED_FILE, CRED_FILE + '.lock'):
            if os.path.exists(p):
                os.remove(p)

    status, repo = api('GET', f'/repos/{full}', token)
    print('done. repo:', repo.get('html_url'), '| size KB:', repo.get('size'))


if __name__ == '__main__':
    main()
