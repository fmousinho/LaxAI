#!/usr/bin/env python3
"""
Scan the repository for accidental duplicate files (e.g., names ending with " 2.py")
and move them into backup_removed_files/ preserving directory structure.

Usage:
  python scripts/cleanup_duplicates.py [--dry-run]

Notes:
  - Excludes .git, __pycache__, .pytest_cache, wandb, artifacts, and backup_removed_files
  - Patterns handled by default: filenames ending with " 2" before the extension (e.g., "foo 2.py").
  - You can extend `DUPLICATE_SUFFIXES` or `EXCLUDE_DIRS` if needed.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = REPO_ROOT / "backup_removed_files"

# Directories to exclude from scanning entirely
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "wandb",
    "artifacts",
    str(BACKUP_ROOT.relative_to(REPO_ROOT)),
}

# Regex patterns for duplicate-style filenames
# Currently: space then the number 2 before an optional extension, e.g., "file 2.py", "README 2.md"
DUPLICATE_PATTERNS = [
    re.compile(r"^.+\s2(\..+)?$"),
]


def is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    return any(excl in parts for excl in EXCLUDE_DIRS)


def is_duplicate_name(path: Path) -> bool:
    name = path.name
    return any(pat.match(name) for pat in DUPLICATE_PATTERNS)


def iter_candidate_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter excluded directories in-place to prevent walking into them
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        current_dir = Path(dirpath)
        if is_excluded(current_dir):
            continue
        for filename in filenames:
            file_path = current_dir / filename
            if is_duplicate_name(file_path):
                yield file_path


def move_to_backup(file_path: Path, dry_run: bool = False):
    rel_path = file_path.relative_to(REPO_ROOT)
    backup_target = BACKUP_ROOT / rel_path
    backup_target.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"DRY-RUN: would move '{rel_path}' -> '{backup_target.relative_to(REPO_ROOT)}'")
    else:
        print(f"Moving '{rel_path}' -> '{backup_target.relative_to(REPO_ROOT)}'")
        shutil.move(str(file_path), str(backup_target))


def main():
    parser = argparse.ArgumentParser(description="Move duplicate-style files to backup_removed_files/")
    parser.add_argument("--dry-run", action="store_true", help="List actions without making changes")
    args = parser.parse_args()

    candidates = list(iter_candidate_files(REPO_ROOT))
    if not candidates:
        print("No duplicate-style files found.")
        return

    print(f"Found {len(candidates)} duplicate-style files.")
    for fp in candidates:
        move_to_backup(fp, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry-run complete. Re-run without --dry-run to apply changes.")
    else:
        print("Done. Files moved into backup_removed_files/.")


if __name__ == "__main__":
    main()
