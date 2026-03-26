import os
from pathlib import Path
from os.path import dirname

STORE_DIR = Path(dirname(dirname(dirname(__file__))), "store")


def _ensure_dir():
    if not os.path.exists(STORE_DIR):
        os.makedirs(STORE_DIR, exist_ok=True)


_ensure_dir()


def file_reducer(left, right):
    merged = []
    if left is None:
        left = []
    if right is None:
        right = []
    for i in left + right:
        if isinstance(i, str):
            merged.append(i)
        if isinstance(i, tuple):
            assert len(i) == 2
            filename, content = i
            merged.append(filename)
            with open(os.path.join(STORE_DIR, filename), "w", encoding='utf-8') as f:
                f.write(content)
    return merged
