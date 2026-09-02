#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}
MAXIMUM_TEXT_BYTES = 64 * 1024


def inspect_file(model_directory: Path, path: Path) -> dict:
    entry = {
        "path": path.relative_to(model_directory).as_posix(),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= MAXIMUM_TEXT_BYTES:
        entry["content"] = path.read_text(encoding="utf-8", errors="replace")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a downloaded mobile speech model package for a coding agent."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", type=Path, default=Path("model-summary.json"))
    args = parser.parse_args()

    model_directory = args.model_dir.resolve()
    if not model_directory.is_dir():
        raise SystemExit(f"Model directory not found: {model_directory}")

    files = [
        inspect_file(model_directory, path)
        for path in sorted(model_directory.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(model_directory).parts)
    ]
    summary = {
        "model_id": args.model_id,
        "runtime": args.runtime,
        "model_directory": str(model_directory),
        "files": files,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
