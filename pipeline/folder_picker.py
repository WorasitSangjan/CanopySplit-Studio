# SPDX-License-Identifier: AGPL-3.0-only

"""Open the platform folder dialog in a separate process."""

import json
import subprocess
import sys
from pathlib import Path

MAC_OPEN_PANEL = """
ObjC.import("AppKit");

function run(argv) {
    const app = $.NSApplication.sharedApplication;
    app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);

    const panel = $.NSOpenPanel.openPanel;
    panel.canChooseFiles = false;
    panel.canChooseDirectories = true;
    panel.allowsMultipleSelection = false;
    panel.canCreateDirectories = argv[3] === "output";
    panel.directoryURL = $.NSURL.fileURLWithPath(argv[0]);
    panel.message = argv[1];
    panel.prompt = argv[2];
    app.activateIgnoringOtherApps(true);
    panel.makeKeyAndOrderFront(null);

    const selected = panel.runModal === $.NSModalResponseOK;
    return JSON.stringify(selected ? ObjC.unwrap(panel.URL.path) : "");
}
"""


def choose_folder(initial, *, purpose="dataset"):
    """Return the chosen path, an empty string on cancel, or None on an unsupported platform."""
    title, button = {
        "dataset": ("Choose the CanopySplit dataset folder", "Choose"),
        "output": ("Choose where to save CanopySplit results", "Save Here"),
        "saved": ("Choose a saved CanopySplit results folder", "Open Results"),
    }[purpose]
    if sys.platform == "darwin":
        command = ["/usr/bin/osascript", "-l", "JavaScript", "-", str(initial), title, button, purpose]
        script = MAC_OPEN_PANEL
    elif sys.platform == "win32":
        command = [sys.executable, "-B", str(Path(__file__).resolve()), str(initial), title]
        script = None
    else:
        return None
    result = subprocess.run(command, input=script, capture_output=True, encoding="utf-8",
                            errors="replace", check=True, timeout=600)
    return json.loads(result.stdout)


def windows_dialog():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    try:
        path = filedialog.askdirectory(
            parent=root, title=sys.argv[2],
            initialdir=sys.argv[1], mustexist=True,
        )
        print(json.dumps(path or ""))
    finally:
        root.destroy()


if __name__ == "__main__":
    windows_dialog()
