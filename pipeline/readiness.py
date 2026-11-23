# SPDX-License-Identifier: AGPL-3.0-only

"""Check the local requirements for installation and measurement."""

import argparse
from contextlib import redirect_stdout
from importlib import import_module, metadata
import json
from pathlib import Path
import sys
from zipfile import BadZipFile, ZipFile

if __package__:
    from .inputs import DEFAULT_WEIGHTS
else:
    from inputs import DEFAULT_WEIGHTS

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def installed_checks():
    version = ".".join(map(str, sys.version_info[:3]))
    checks = [{"name": "Python", "ok": sys.version_info >= (3, 10),
               "detail": f"Python {version}. Version 3.10 or newer is required."}]
    problems = []
    try:
        from packaging.requirements import Requirement
    except ImportError:
        problems.append("The packaging library is missing.")
    else:
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            text = line.split(" #", 1)[0].strip()
            if not text or text.startswith("#"):
                continue
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                problems.append(f"{requirement.name} is missing.")
            else:
                if not requirement.specifier.contains(installed):
                    problems.append(f"{requirement.name} {installed} needs {requirement.specifier}.")
    checks.append({"name": "Required libraries", "ok": not problems,
                   "detail": " ".join(problems) or "All required versions are installed."})
    return checks


def model_check():
    try:
        with ZipFile(DEFAULT_WEIGHTS) as model:
            available = any(name.rsplit("/", 1)[-1] == "data.pkl" for name in model.namelist())
    except (OSError, BadZipFile):
        available = False
    detail = ("The bundled YOLO weights are available." if available else
              "The bundled YOLO weights are missing or incomplete. Unzip a complete copy of CanopySplit Studio.")
    return {"name": "Pot detector model", "ok": available, "detail": detail}


def runtime_check():
    name = "Image processing and detection"
    try:
        for module in ("cv2", "numpy", "scipy", "PIL", "matplotlib"):
            import_module(module)
        import torch
        from torchvision.ops import batched_nms
        from ultralytics import YOLO

        if not callable(YOLO):
            raise TypeError("The YOLO model loader is unavailable.")
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        batched_nms(boxes, torch.tensor([1.0]), torch.tensor([0]), 0.5)
    except Exception as error:
        return {"name": name, "ok": False, "detail": f"A required library could not start. {error}"}
    return {"name": name, "ok": True, "detail": "The image and detection libraries loaded successfully."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("installed", "runtime"))
    args = parser.parse_args()
    if args.mode == "runtime":
        with redirect_stdout(sys.stderr):
            check = runtime_check()
        print(json.dumps(check))
        return 0
    problems = [check["detail"] for check in installed_checks() if not check["ok"]]
    for problem in problems:
        print(problem, file=sys.stderr)
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
