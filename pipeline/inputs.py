# SPDX-License-Identifier: AGPL-3.0-only

"""Photo naming, tables and portable paths shared by the app and pipeline."""

import csv
from hashlib import sha256
import re
from pathlib import Path

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "pot_detector_yolo26s.pt"
PHOTO_NAME = re.compile(r"day(?P<day>\d+)_(?P<channel>RGB|NIR)_(?P<tray>.+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def file_digest(path):
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(folder):
    """Supported images directly inside the selected folder."""
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")
                  and not path.name.startswith(".") and path.is_file())


def photo_files(folder, channel):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and (match := PHOTO_NAME.fullmatch(path.name))
        and match["channel"].upper() == channel
    )


def saved_path(settings, key, results):
    """Resolve local paths and accept relative paths written on Windows."""
    absolute = settings.get(key)
    relative = settings.get(f"{key}_rel")
    original = Path(absolute).expanduser().resolve() if absolute else None
    if relative:
        local = (results / relative.replace("\\", "/")).resolve()
        expected = settings.get("rgb_inputs") if key == "photos" else None
        if expected and local != original and local.is_dir() and original is not None and original.is_dir():
            names = [name for name, _ in expected]
            if not all((local / name).is_file() for name in names) and all((original / name).is_file() for name in names):
                return original
        if local.exists() or not absolute:
            return local
    return original


def photo_key(match):
    return int(match["day"]), match["tray"].casefold()


def group_labels(names):
    """Use the spelling from the first day for each case-insensitive group."""
    matches = [match for name in names if (match := PHOTO_NAME.fullmatch(name))]
    labels = {}
    for match in sorted(matches, key=photo_key):
        label = match["tray"]
        labels.setdefault(label.casefold(), label)
    return labels
