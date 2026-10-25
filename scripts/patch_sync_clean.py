# -*- coding: utf-8 -*-
"""Patch sync_github_staging.py so it never carries large binaries/checkpoints into the staging repo."""
import io

P = r'E:\FINO\scripts\sync_github_staging.py'
s = io.open(P, encoding='utf-8').read()

# add generic cleaner after clean_gpu_exp
anchor = 'def trim_logs():'
assert anchor in s
cleaner = '''def clean_run_dir(rel):
    """Keep only CSV/JSON under a run-output directory (drop binary checkpoints)."""
    root = os.path.join(DST, rel)
    if not os.path.isdir(root):
        return
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1] not in ('.csv', '.json'):
                os.remove(os.path.join(dirpath, f))
    for dirpath, dirs, files in os.walk(root, topdown=False):
        if dirpath != root and not os.listdir(dirpath):
            os.rmdir(dirpath)


def clean_wsl_binaries():
    root = os.path.join(DST, 'wsl_setup')
    if not os.path.isdir(root):
        return
    for f in os.listdir(root):
        if f.endswith(('.tar.gz', '.msi')):
            os.remove(os.path.join(root, f))


'''
s = s.replace(anchor, cleaner + anchor, 1)

# call cleaners after copies
old = """    copy_tree('gpu_exp')
    clean_gpu_exp()
    trim_logs()"""
new = """    copy_tree('gpu_exp')
    clean_gpu_exp()
    clean_run_dir('exp')
    clean_run_dir('exp_restore')
    clean_wsl_binaries()
    trim_logs()"""
assert old in s
s = s.replace(old, new, 1)

io.open(P, 'w', encoding='utf-8').write(s)
print('sync script patched')
