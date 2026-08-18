#!/usr/bin/env python3

import argparse
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from huggingface_hub import hf_hub_download, snapshot_download


LITERT_REPOS = {
    "Arm/whisper-base-int8-litert",
    "Arm/whisper-medium-int8-litert",
    "Arm/whisper-large-v3-int8-litert",
}

EXECUTORCH_REPOS = {
    "Arm/whisper-tiny-int8-xnnpack-executorch": {
        "model": "whisper-tiny-int8-executorch.pte",
        "preprocessor": "whisper-tiny-preprocessor-int8-executorch.pte",
        "tokenizer_repo": "openai/whisper-tiny",
    },
    "Arm/whisper-small-int8-xnnpack-executorch": {
        "model": "whisper-small-int8-executorch.pte",
        "preprocessor": "whisper-small-preprocessor-int8-executorch.pte",
        "tokenizer_repo": "openai/whisper-small",
    },
}

LITERT_PACKAGE_FILES = (
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

PACKAGE_METADATA = (
    "config.yaml",
    "metadata.yaml",
)


def package_files(archive: Path, files: list[tuple[Path, str]]) -> None:
    with ZipFile(archive, "w", compression=ZIP_STORED, allowZip64=True) as package:
        for path, archive_name in files:
            package.write(path, archive_name)


def collect_litert_package(snapshot_dir: Path) -> list[tuple[Path, str]]:
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

    return [
        (path, path.name)
        for path in sorted(snapshot_dir.rglob("*"))
        if path.is_file()
        and ".cache" not in path.relative_to(snapshot_dir).parts
        and path.name != ".gitignore"
    ]


def collect_executorch_package(
    repo_id: str,
    snapshot_dir: Path,
    output_dir: Path,
) -> list[tuple[Path, str]]:
    package = EXECUTORCH_REPOS[repo_id]
    model = snapshot_dir / package["model"]
    preprocessor = snapshot_dir / package["preprocessor"]
    if not model.is_file():
        raise SystemExit(f"Missing ExecuTorch model: {model}")
    if not preprocessor.is_file():
        raise SystemExit(f"Missing ExecuTorch preprocessor: {preprocessor}")

    tokenizer_repo = package["tokenizer_repo"]
    tokenizer_dir = output_dir / tokenizer_repo.replace("/", "__")
    tokenizer = Path(
        hf_hub_download(
            repo_id=tokenizer_repo,
            filename="tokenizer.json",
            local_dir=tokenizer_dir,
        )
    )

    files = [
        (model, model.name),
        (preprocessor, "whisper_preprocessor.pte"),
        (tokenizer, "tokenizer.json"),
    ]
    for name in PACKAGE_METADATA:
        path = snapshot_dir / name
        if path.is_file():
            files.append((path, name))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and package a Whisper model for Whisper Journal."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--output-dir", default="models", help="Download/package directory")
    args = parser.parse_args()

    supported_repos = LITERT_REPOS | set(EXECUTORCH_REPOS)
    if args.repo_id not in supported_repos:
        supported = "\n  ".join(sorted(supported_repos))
        raise SystemExit(f"Unsupported repository ID. Choose one of:\n  {supported}")

    output_dir = Path(args.output_dir)
    snapshot_dir = output_dir / args.repo_id.replace("/", "__")
    if args.repo_id in LITERT_REPOS:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=snapshot_dir,
            allow_patterns=list(LITERT_PACKAGE_FILES),
        )
        files = collect_litert_package(snapshot_dir)
    else:
        package = EXECUTORCH_REPOS[args.repo_id]
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=snapshot_dir,
            allow_patterns=[
                package["model"],
                package["preprocessor"],
                *PACKAGE_METADATA,
            ],
        )
        files = collect_executorch_package(args.repo_id, snapshot_dir, output_dir)

    archive = output_dir / f"{args.repo_id.split('/')[-1]}.zip"
    package_files(archive, files)

    print(f"Model directory: {snapshot_dir.resolve()}")
    print(f"Android package: {archive.resolve()}")


if __name__ == "__main__":
    main()
