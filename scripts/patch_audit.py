# -*- coding: utf-8 -*-
"""Fix blob() in audit_github_repo.py to support upstream repo blobs."""
import io

P = r'E:\FINO\scripts\audit_github_repo.py'
s = io.open(P, encoding='utf-8').read()
old1 = "def blob(sha):\n    r = get(f'/repos/engulf408/FINO/git/blobs/{sha}')"
new1 = "def blob(sha, repo='engulf408/FINO'):\n    r = get(f'/repos/{repo}/git/blobs/{sha}')"
assert old1 in s
s = s.replace(old1, new1, 1)
old2 = "up_text = blob(up_sha).decode('utf-8', 'replace') if up_sha else None"
new2 = "up_text = blob(up_sha, 'CTID282/FINO').decode('utf-8', 'replace') if up_sha else None"
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('audit script fixed')
