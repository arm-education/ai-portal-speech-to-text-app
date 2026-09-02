#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


MODEL_SUFFIXES = {".onnx", ".pte", ".pt", ".pth", ".tflite"}
FILENAME_PATTERN = re.compile(r"(?m)^filename\s*:\s*([^\s#]+)")


def model_directory(output_directory: Path, model_id: str) -> Path:
    return output_directory / model_id.replace("/", "__")


def primary_model_path(directory: Path) -> Path:
    for metadata_name in ("metadata.yaml", "metadata.yml"):
        metadata_path = directory / metadata_name
        if not metadata_path.is_file():
            continue
        match = FILENAME_PATTERN.search(
            metadata_path.read_text(encoding="utf-8", errors="replace")
        )
        if not match:
            continue
        filename = match.group(1).strip().strip(chr(39) + chr(34))
        candidate = (directory / filename).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as exception:
            raise SystemExit(
                f"The model filename in {metadata_path} is outside the package directory."
            ) from exception
        if candidate.is_file():
            return candidate
        raise SystemExit(
            f"The primary model file recorded in {metadata_path} was not found: {candidate}"
        )

    candidates = [
        path.resolve()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit("The downloaded package does not identify a model binary.")
    raise SystemExit(
        "The downloaded package contains multiple model binaries and does not identify "
        "the primary file in metadata.yaml."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a complete speech model package for adapter generation."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--print-model-path",
        action="store_true",
        help="Print the primary model binary instead of the package directory.",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    destination = model_directory(args.output_dir, args.repo_id)
    downloaded = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=destination,
    )
    downloaded_directory = Path(downloaded).resolve()
    if args.print_model_path:
        print(primary_model_path(downloaded_directory))
    else:
        print(downloaded_directory)


if __name__ == "__main__":
    main()
