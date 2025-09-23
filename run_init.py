# run_init.py
from pathlib import Path
from src.pdf2anki.cli import app

if __name__ == "__main__":
    # Pick a target directory (here: ./test_project)
    target_dir = Path("./")

    # Equivalent to running: pdf2anki init test_project --force
    if False:
        try:
            app(["init", str(target_dir), "--force"])
        except SystemExit as e:
            # Typer/Click exits with SystemExit
            print(f"Exited with code {e.code}")

    if True:
        try:
            import os
            os.chdir(target_dir)
            app(["generate", "--config", f"config.yaml", "--sample"])
            # app(["generate",  "--config", f"{target_dir}/config.yaml", "--documents", f"{target_dir}/documents.yaml", "--plan"])
        except SystemExit as e:
            # Typer/Click exits with SystemExit
            print(f"Exited with code {e.code}")
