# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 3b: combine automatic leaf masks with manual corrections.

For every photo, masks_final/<photo>/ gets the corrected masks if the photo
was saved in the correction tool (corrections/<photo>/done.json), otherwise
the automatic masks. If a corrected photo lacks a mask for some pot (for
example a pot that was added to the workflow after the correction was
saved), that pot gets its automatic mask minus the pixels the person already
gave to other plants, so manual decisions always take priority.

The automatic masks are never changed, so re-running the automatic workflow
keeps all manual work.

Usage:
    python apply_corrections.py --auto masks/ --corrections corrections/ --out masks_final/
"""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
if __package__:
    from .image_io import read_image, write_image
    from .outputs import publish_outputs
else:
    from image_io import read_image, write_image
    from outputs import publish_outputs



def main():
    ap = argparse.ArgumentParser(description="Apply manual corrections on top of automatic masks")
    ap.add_argument("--auto", required=True)
    ap.add_argument("--corrections", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    auto, corr, out = Path(a.auto), Path(a.corrections), Path(a.out)
    if out.is_symlink():
        raise SystemExit("Choose a mask output folder rather than a symbolic link.")
    auto, corr, out = (path.resolve() for path in (auto, corr, out))
    if not auto.is_dir():
        raise SystemExit(f"Automatic mask folder not found: {auto}")
    if any(out == source or out in source.parents or source in out.parents for source in (auto, corr)):
        raise SystemExit("Choose an output folder separate from the automatic and corrected masks.")
    if out.exists() and (not out.is_dir() or any(
        not folder.is_dir() or folder.is_symlink()
        or any(not mask.is_file() or not mask.match("pot_*_mask.png") for mask in folder.iterdir())
        for folder in out.iterdir()
    )):
        raise SystemExit("The mask output folder contains additional files. Choose a separate mask output folder.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".masks-", dir=out.parent) as temporary:
        staged = Path(temporary)
        work = staged / out.name
        work.mkdir()
        n_manual, n_filled, total = apply_masks(auto, corr, work)
        publish_outputs(staged, out.parent, (out.name,))
    print(f"{n_manual} photos with manual corrections ({n_filled} pot masks filled from the automatic result), "
          f"{total - n_manual} automatic -> {out}")


def apply_masks(auto, corr, out):
    n_manual = n_filled = 0
    photos = sorted(p for p in auto.iterdir() if p.is_dir())
    for d in photos:
        c = corr / d.name
        if not (c / "done.json").exists():
            shutil.copytree(d, out / d.name)
            continue
        n_manual += 1
        shutil.copytree(c, out / d.name, ignore=shutil.ignore_patterns("done.json"))
        missing = [m for m in d.glob("pot_*_mask.png") if not (c / m.name).exists()]
        if missing:
            taken = None
            for m in c.glob("pot_*_mask.png"):
                gray = read_image(str(m), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise OSError(f"Cannot read mask: {m}")
                x = gray > 0
                taken = x if taken is None else (taken | x)
            for m in missing:
                gray = read_image(str(m), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise OSError(f"Cannot read mask: {m}")
                x = gray > 0
                if taken is not None:
                    x &= ~taken
                path = out / d.name / m.name
                if not write_image(str(path), x.astype(np.uint8) * 255):
                    raise OSError(f"Could not write final mask: {path}")
                n_filled += 1
    return n_manual, n_filled, len(photos)


if __name__ == "__main__":
    main()
