# -*- coding: utf-8 -*-
"""Sync updated workspace into the GitHub staging repo and commit locally."""
import os
import shutil
import subprocess

SRC = r'E:\FINO'
DST = r'E:\FINO\_gh\FINO-reproduction'
GIT = r'C:\Program Files\Git\cmd\git.exe'


def copy_tree(rel, exclude_names=()):
    src = os.path.join(SRC, rel)
    dst = os.path.join(DST, rel)
    if not os.path.isdir(src):
        return 0
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in exclude_names]
        r = os.path.relpath(root, src)
        out = dst if r == '.' else os.path.join(dst, r)
        os.makedirs(out, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(out, f))
            n += 1
    return n


def copy_file(rel):
    src, dst = os.path.join(SRC, rel), os.path.join(DST, rel)
    os.makedirs(os.path.dirname(dst) or DST, exist_ok=True)
    shutil.copy2(src, dst)


def clean_gpu_exp():
    root = os.path.join(DST, 'gpu_exp')
    if not os.path.isdir(root):
        return
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1] not in ('.csv', '.json'):
                os.remove(os.path.join(dirpath, f))
    for dirpath, dirs, files in os.walk(root, topdown=False):
        if dirpath != root and not os.listdir(dirpath):
            os.rmdir(dirpath)


def trim_logs():
    src_root = os.path.join(SRC, 'logs')
    dst_root = os.path.join(DST, 'logs')
    for root, dirs, files in os.walk(src_root):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, src_root)
            out = os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(out) or dst_root, exist_ok=True)
            size = os.path.getsize(p)
            if size < 500_000:
                shutil.copy2(p, out)
                continue
            lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
            keep = [l for l in lines if not ('it/s]' in l and '%|' in l)]
            head, tail = keep[:200], keep[-200:]
            with open(out, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(head + [f'... [trimmed {len(lines)-len(head)-len(tail)} progress lines] ...'] + tail) + '\n')


def main():
    n = 0
    n += copy_tree('FINO', exclude_names={'__pycache__', 'wandb'})
    n += copy_tree('docs')
    n += copy_tree('scripts')
    n += copy_tree('wsl_setup')
    n += copy_tree('results')
    n += copy_tree('ppt', exclude_names={})
    n += copy_tree('experiments', exclude_names={'__pycache__'})
    n += copy_tree('exp')
    n += copy_tree('exp_restore')
    copy_file('README.md')
    copy_file('constraints.txt')
    copy_file('.gitignore')
    copy_tree('gpu_exp')
    clean_gpu_exp()
    trim_logs()
    # remove ppt preview temp dirs / big files if any
    ppt_dir = os.path.join(DST, 'ppt')
    for d in os.listdir(ppt_dir):
        if d.startswith('.dsh-office-'):
            shutil.rmtree(os.path.join(ppt_dir, d), ignore_errors=True)
    # update .gitignore in staging
    gi = os.path.join(DST, '.gitignore')
    txt = open(gi, encoding='utf-8').read() if os.path.exists(gi) else ''
    if 'gpu_exp' not in txt:
        txt += ('\n# GPU 运行：保留 CSV/flags，排除二进制 checkpoint\ngpu_exp/**/1000000/\n'
                'ppt/.dsh-office-*/\n')
        open(gi, 'w', encoding='utf-8').write(txt)
    print('synced files:', n)
    # git commit
    subprocess.run([GIT, '-C', DST, 'add', '-A'], check=True)
    msg = ('Add full-budget GPU reproduction results (WSL2 + RTX 5070): '
           'antmaze-large 1M+500k FINO 100% vs FQL 92%, GPU benchmark/analysis scripts, '
           'WSL setup scripts, updated report and PPT')
    r = subprocess.run([GIT, '-C', DST, 'commit', '-q', '-m', msg], capture_output=True, text=True)
    print('git commit rc=', r.returncode, r.stdout.strip(), r.stderr.strip()[:300])
    subprocess.run([GIT, '-C', DST, 'log', '--oneline', '-3'])
    st = subprocess.run([GIT, '-C', DST, 'status', '--short'], capture_output=True, text=True)
    print('status short:', st.stdout.strip()[:800])


if __name__ == '__main__':
    main()
