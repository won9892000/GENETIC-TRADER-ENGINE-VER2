"""Cleanup helper to remove generated artifacts: __pycache__, *.pyc, and optional old run dirs.

Usage:
  python tools/cleanup.py --pyc --runs --max-age-days 7
"""
import argparse
from pathlib import Path
import shutil
import time

ROOT = Path('.').resolve()

def remove_pyc(root=ROOT):
    removed = []
    for p in root.rglob('*.pyc'):
        try:
            p.unlink()
            removed.append(str(p))
        except Exception:
            pass
    for d in root.rglob('__pycache__'):
        try:
            shutil.rmtree(d)
            removed.append(str(d))
        except Exception:
            pass
    return removed


def remove_old_runs(root=ROOT, max_age_days=7):
    removed = []
    now = time.time()
    runs_dir = root / 'runs'
    if not runs_dir.exists():
        return removed
    for child in runs_dir.iterdir():
        try:
            mtime = child.stat().st_mtime
            age_days = (now - mtime) / 86400.0
            if age_days > max_age_days:
                if child.is_dir():
                    shutil.rmtree(child)
                    removed.append(str(child))
                else:
                    child.unlink()
                    removed.append(str(child))
        except Exception:
            pass
    return removed

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pyc', action='store_true')
    p.add_argument('--runs', action='store_true')
    p.add_argument('--max-age-days', type=int, default=7)
    args = p.parse_args()
    if args.pyc:
        r = remove_pyc()
        print('Removed pyc/__pycache__:', len(r))
    if args.runs:
        r = remove_old_runs(max_age_days=args.max_age_days)
        print('Removed old runs:', len(r))
