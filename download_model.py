#!/usr/bin/env python3

import argparse
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from huggingface_hub import snapshot_download


PACKAGE_FILES = (
    "*int8*.tflite",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "normalizer.json",
    "config.yaml",
    "metadata.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and package a LiteRT Whisper model for Whisper Journal."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--output-dir", default="models", help="Download/package directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / args.repo_id.replace("/", "__")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=snapshot_dir,
        allow_patterns=list(PACKAGE_FILES),
    )

    tflite_files = sorted(snapshot_dir.rglob("*.tflite"))
    tokenizer_files = sorted(snapshot_dir.rglob("tokenizer.json"))
    if len(tflite_files) != 1:
        raise SystemExit(
            f"Expected one .tflite file under {snapshot_dir}, found {len(tflite_files)}"
        )
    if len(tokenizer_files) != 1:
        raise SystemExit(
            f"Expected one tokenizer.json under {snapshot_dir}, found {len(tokenizer_files)}"
        )

    archive = output_dir / f"{args.repo_id.split('/')[-1]}.zip"
    files = [
        path
        for path in snapshot_dir.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(snapshot_dir).parts
        and path.name != ".gitignore"
    ]
    with ZipFile(archive, "w", compression=ZIP_STORED, allowZip64=True) as package:
        for path in sorted(files):
            package.write(path, path.name)

    print(f"Model directory: {snapshot_dir.resolve()}")
    print(f"Android package: {archive.resolve()}")


if __name__ == "__main__":
    main()
