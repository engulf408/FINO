# -*- coding: utf-8 -*-
"""Add the course/ directory to the GitHub staging sync."""
import io

P = r'E:\FINO\scripts\sync_github_staging.py'
s = io.open(P, encoding='utf-8').read()
if "copy_tree('course')" in s:
    print('already patched')
else:
    old = "    copy_tree('gpu_exp')\n    clean_gpu_exp()"
    new = ("    copy_tree('gpu_exp')\n    copy_tree('course')\n"
           "    course_dir = os.path.join(DST, 'course')\n"
           "    if os.path.isdir(course_dir):\n"
           "        for d in os.listdir(course_dir):\n"
           "            if d.startswith('.dsh-office-'):\n"
           "                shutil.rmtree(os.path.join(course_dir, d), ignore_errors=True)\n"
           "    clean_gpu_exp()")
    assert old in s
    s = s.replace(old, new, 1)
    io.open(P, 'w', encoding='utf-8').write(s)
    print('sync script patched with course/')
