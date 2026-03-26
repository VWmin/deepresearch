import os
from pathlib import Path
from os.path import dirname

STORE_DIR = Path(dirname(dirname(dirname(__file__))), "store")


def _ensure_dir():
    if not os.path.exists(STORE_DIR):
        os.makedirs(STORE_DIR, exist_ok=True)


_ensure_dir()


def file_reducer(left: list[str], right: list[tuple[str, str]]):
    if left is None:
        left = []
    if right is None:
        return left
    for filename, content in right:
        if filename not in left:
            left.append(filename)
        with open(os.path.join(STORE_DIR, filename), "w") as f:
            f.write(content)
    return left
