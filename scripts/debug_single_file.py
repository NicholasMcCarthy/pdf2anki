#!/usr/bin/env python3
"""Debug helper: run exactly the same per-file processing pdf2anki serve does
(src/pdf2anki/service/runner.py's process_new_file()) against a single local
file, with full tracebacks - no documents.yaml/scan-docs needed, since the
watcher service never uses that (it classifies + builds settings on the fly).

Usage:
    python debug_single_file.py path/to/file.pdf
    python debug_single_file.py path/to/file.md

Reuses your real config.yml as-is (same LLM provider/model/strategies as
docker-compose), just points output paths at ./debug-workspace locally
instead of the container's /home/pdf2anki/workspace.
"""
import sys
from pathlib import Path

# --- Load .env into the environment, same values docker-compose's env_file uses ---
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os_environ_key = key.strip()
        if os_environ_key and os_environ_key not in __import__("os").environ:
            __import__("os").environ[os_environ_key] = value.strip()

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from pdf2anki.config import Config
from pdf2anki.service.runner import process_new_file

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <path-to-pdf-or-md>")
    sys.exit(1)

target = Path(sys.argv[1]).resolve()
if not target.exists():
    print(f"File not found: {target}")
    sys.exit(1)

config = Config.from_yaml("config.yml")

# Override container-absolute output paths with local ones so this doesn't
# try to write into /home/pdf2anki/workspace, which doesn't exist here.
workspace = Path("./debug-workspace").resolve()
config.generate.output.workspace = workspace
config.generate.output.csv_path = workspace / "cards.csv"
config.generate.output.media_path = workspace / "media"
config.generate.output.apkg_path = workspace / "deck.apkg"
config.generate.output.manifest_path = workspace / "manifest.json"

print(f"Processing {target} ...")
result = process_new_file(target, config)

print("\n--- Result ---")
for key, value in result.items():
    print(f"{key}: {value}")
