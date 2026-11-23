#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Double-click in Finder to set up and open CanopySplit Studio.
set -euo pipefail

setup_dir=
finish() {
  status=$?
  if [[ -n "$setup_dir" ]]; then
    /bin/rm -rf -- "$setup_dir"
  fi
  if [[ "$status" -ne 0 && -t 0 ]]; then
    printf '\nCanopySplit Studio could not start. You can retry by opening this file again.\n'
    read -r -p 'Press Return to close this window. ' || true
  fi
}
trap finish EXIT

cd -- "$(dirname -- "$0")"
python=.venv/bin/python
printf 'Checking CanopySplit Studio setup on this Mac.\n'
if [[ ! -x "$python" ]] || ! "$python" -B -m pipeline.readiness installed; then
  if [[ "$(uname -m)" != arm64 ]]; then
    printf 'Automatic setup requires an Apple silicon Mac and a native Terminal session.\n' >&2
    exit 1
  fi
  if ! permission=$(/usr/bin/osascript <<'APPLESCRIPT'
set setupMessage to "CanopySplit Studio needs Python or required libraries before it can open." & return & return
set setupMessage to setupMessage & "Choose Install to download the required software and set up the local app environment. Setup needs an internet connection."
try
    set response to display dialog setupMessage with title "Set up CanopySplit Studio" buttons {"Cancel", "Install"} default button "Cancel" cancel button "Cancel"
    return button returned of response
on error number -128
    return "Cancel"
end try
APPLESCRIPT
  ); then
    printf 'Could not show the setup prompt. Open start_mac.command from Finder to try again.\n' >&2
    exit 1
  fi
  if [[ "$permission" != Install ]]; then
    printf 'Setup canceled. Open start_mac.command again whenever you are ready.\n'
    exit 0
  fi
  printf 'Setting up CanopySplit Studio. Keep this window open while Python and the required packages are installed.\n'

  app_data="$PWD/.canopysplit"
  mkdir -p -- "$app_data"
  export UV_PYTHON_INSTALL_DIR="$app_data/python"
  export UV_CACHE_DIR="$app_data/uv-cache"
  export UV_PYTHON_INSTALL_BIN=0

  # Check the official uv release archive before running its installer binary.
  uv_version=0.13.9
  uv_target=uv-aarch64-apple-darwin
  uv_sha256=5aeca67f2283f75235de40d0bd9c0a5bbc5fce23b6d48b3bf9fa1b2db08382ca
  setup_dir=$(mktemp -d "$app_data/setup.XXXXXX")
  archive="$setup_dir/$uv_target.tar.gz"
  /usr/bin/curl --fail --silent --show-error --location --proto '=https' \
    --retry 2 --connect-timeout 15 --output "$archive" \
    "https://github.com/astral-sh/uv/releases/download/$uv_version/$uv_target.tar.gz"
  if ! printf '%s  %s\n' "$uv_sha256" "$archive" | /usr/bin/shasum -a 256 --check --status; then
    printf 'The setup download could not be verified. Please open this file again to retry.\n' >&2
    exit 1
  fi
  /usr/bin/tar -xzf "$archive" -C "$setup_dir" "$uv_target/uv"
  uv="$setup_dir/$uv_target/uv"

  if [[ ! -x "$python" ]] || ! "$python" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    "$uv" venv --no-config --allow-existing --managed-python --python 3.11 .venv
  fi
  "$uv" pip install --no-config --python "$python" -r requirements.txt
  "$uv" pip check --no-config --python "$python"
  "$python" -B -m pipeline.readiness installed
  /bin/rm -rf -- "$setup_dir"
  setup_dir=
fi

printf 'Opening CanopySplit Studio in your browser. Keep this window open while you work.\n'
"$python" -B -u app.py --port 0 "$@"
