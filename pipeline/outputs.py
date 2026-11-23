# SPDX-License-Identifier: AGPL-3.0-only

"""Publish completed outputs from a temporary directory on the same volume."""

from pathlib import Path
import shutil
from tempfile import mkdtemp

from filelock import FileLock


def results_lock(destination):
    return FileLock(destination / ".canopysplit.lock", timeout=0)


def publish_outputs(staged, destination, names):
    for name in names:
        if not (staged / name).exists():
            raise FileNotFoundError(f"Missing completed output: {name}")
    backup = Path(mkdtemp(prefix=".previous-", dir=destination))
    moved, published = [], []
    try:
        for name in names:
            target = destination / name
            if target.exists() or target.is_symlink():
                target.replace(backup / name)
                moved.append(name)
            (staged / name).replace(target)
            published.append(name)
    except OSError:
        try:
            for name in reversed(published):
                (destination / name).replace(staged / name)
            for name in reversed(moved):
                (backup / name).replace(destination / name)
        except OSError as recovery_error:
            raise OSError(f"Could not restore all previous outputs. Remaining backups are in {backup}.") from recovery_error
        shutil.rmtree(backup, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)
