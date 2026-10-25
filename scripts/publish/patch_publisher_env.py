# -*- coding: utf-8 -*-
"""Allow publish_update_api.py to read the token from the GITHUB_TOKEN environment variable."""
import io

P = r'E:\FINO\_gh\publish_update_api.py'
s = io.open(P, encoding='utf-8').read()
old = "    ap.add_argument('--token-file', required=True)"
new = "    ap.add_argument('--token-file', default=None, help='Path to token file; omit to use GITHUB_TOKEN env')\n    ap.add_argument('--message', default='Sync reproduction workspace')"
if new in s:
    print('already patched')
else:
    assert old in s
    s = s.replace(old, new, 1)
    # remove the old --message line if it duplicated (keep single definition)
    dup = ("    ap.add_argument('--token-file', default=None, help='Path to token file; omit to use GITHUB_TOKEN env')\n"
           "    ap.add_argument('--message', default='Sync reproduction workspace')\n")
    s = s.replace(dup, dup, 1)
    old_read = "    token = open(args.token_file, encoding='utf-8').read().strip()"
    new_read = ("    if args.token_file:\n"
                "        token = open(args.token_file, encoding='utf-8').read().strip()\n"
                "    else:\n"
                "        token = os.environ.get('GITHUB_TOKEN', '').strip()\n"
                "    if not token:\n"
                "        sys.exit('no token: pass --token-file or set GITHUB_TOKEN')")
    assert old_read in s
    s = s.replace(old_read, new_read, 1)
    io.open(P, 'w', encoding='utf-8').write(s)
    print('publisher patched for GITHUB_TOKEN env')
