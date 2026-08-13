"""Securely migrate an existing Face++ pair into this project's .env.

The secret values are never printed. The generated .env is excluded by the
project's .gitignore and should remain local to the analysis workstation.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / ".env"


def extract_assignment(source: str, variable: str) -> str:
    pattern = rf"(?m)^{re.escape(variable)}\s*=\s*(['\"])(.*?)\1\s*(?:#.*)?$"
    match = re.search(pattern, source)
    if not match or not match.group(2).strip():
        raise ValueError(f"Could not find a non-empty {variable} assignment.")
    return match.group(2).strip()


def migrate(source_path: Path, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "The output file already exists; use --overwrite to replace it intentionally."
        )
    source = source_path.read_text(encoding="utf-8")
    key = extract_assignment(source, "API_KEY")
    secret = extract_assignment(source, "API_SECRET")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(output_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"FPP_KEY={key}\n")
            handle.write(f"FPP_SECRET={secret}\n")
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    migrate(args.source, args.output, overwrite=args.overwrite)
    print("Face++ credentials migrated (values hidden).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
