"""Robust parallel HTTP Range downloader for large files on slow servers.

Usage: python dl_parallel.py --url URL --out PATH [--chunk-mb 4] [--workers 24]
Writes chunks under <out>.parts/, resumes automatically, verifies total size and
(for .npz) that numpy can open the archive, then atomically renames to PATH.
"""
import argparse
import os
import sys
import time
import urllib.request

RETRIES = 10


def head_length(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def download_range(url, start, end, path):
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            expected = end - start + 1
            if len(data) != expected:
                raise IOError(f'short read {len(data)}/{expected}')
            with open(path + '.tmp', 'wb') as f:
                f.write(data)
            os.replace(path + '.tmp', path)
            return
        except Exception as e:  # noqa: BLE001
            wait = min(60, 2 ** attempt) + (time.time() % 3)
            print(f'  retry chunk {start}-{end} attempt {attempt+1}/{RETRIES}: {e}', flush=True)
            time.sleep(wait)
    raise RuntimeError(f'chunk {start}-{end} failed after {RETRIES} attempts')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk-mb', type=int, default=4)
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f'already exists: {args.out}', flush=True)
        return
    total = head_length(args.url)
    print(f'total={total} bytes', flush=True)
    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(i, min(i + chunk - 1, total - 1)) for i in range(0, total, chunk)]
    parts_dir = args.out + '.parts'
    os.makedirs(parts_dir, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = []
    for i, (s, e) in enumerate(ranges):
        p = os.path.join(parts_dir, f'{i:05d}.bin')
        if os.path.exists(p) and os.path.getsize(p) == e - s + 1:
            continue
        todo.append((i, s, e, p))
    print(f'chunks={len(ranges)} todo={len(todo)} workers={args.workers}', flush=True)

    t0 = time.time()
    done_bytes = sum(os.path.getsize(os.path.join(parts_dir, f'{i:05d}.bin'))
                     for i in range(len(ranges)) if os.path.exists(os.path.join(parts_dir, f'{i:05d}.bin')))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download_range, args.url, s, e, p): (s, e) for _, s, e, p in todo}
            finished = 0
            for fut in as_completed(futs):
                fut.result()
                finished += 1
                s, e = futs[fut]
                done_bytes += e - s + 1
                el = time.time() - t0
                if finished % 5 == 0 or finished == len(todo):
                    print(f'progress {done_bytes/1e6:.1f}/{total/1e6:.1f} MB '
                          f'speed={done_bytes/el/1024:.1f} KB/s elapsed={el:.0f}s', flush=True)
    except Exception as e:  # noqa: BLE001
        print(f'FAILED: {e}', flush=True)
        sys.exit(1)

    # merge in order
    print('merging...', flush=True)
    got = sum(os.path.getsize(os.path.join(parts_dir, f'{i:05d}.bin')) for i in range(len(ranges)))
    if got != total:
        print(f'size mismatch: got={got} expected={total}', flush=True)
        sys.exit(1)
    tmp_out = args.out + '.merging'
    with open(tmp_out, 'wb') as out:
        for i in range(len(ranges)):
            with open(os.path.join(parts_dir, f'{i:05d}.bin'), 'rb') as f:
                out.write(f.read())
    if os.path.getsize(tmp_out) != total:
        print('merged size mismatch', flush=True)
        sys.exit(1)
    if args.out.endswith('.npz'):
        import numpy as np
        with np.load(tmp_out, allow_pickle=False) as f:
            files = list(f.files)
            for k in files:
                _ = f[k].shape
        print(f'npz ok keys={files}', flush=True)
    os.replace(tmp_out, args.out)
    import shutil
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f'DONE {args.out} {total} bytes in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
