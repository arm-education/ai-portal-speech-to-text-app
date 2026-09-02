#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path
from pathlib import PurePosixPath
from zipfile import ZIP_STORED, ZipFile

from huggingface_hub import snapshot_download


VERIFIED_MODELS = {
    "Arm/whisper-base-int8-litert": {
        "runtime": "litert",
        "model": "whisper_base_vivo_litert_optimized.tflite",
        "tokenizer": "tokenizer.json",
    },
    "Arm/whisper-medium-int8-litert": {
        "runtime": "litert",
        "model": "whisper_medium_vivo_litert_optimized.tflite",
        "tokenizer": "tokenizer.json",
    },
    "Arm/whisper-large-v3-int8-litert": {
        "runtime": "litert",
        "model": "whisper_large_v3_vivo_litert_optimized.tflite",
        "tokenizer": "tokenizer.json",
    },
    "Arm/whisper-tiny-int8-xnnpack-executorch": {
        "runtime": "executorch",
        "model": "pte_optimized/whisper_tiny_vivo_executorch_optimized.pte",
        "preprocessor": "pte_optimized/whisper_preprocessor.pte",
        "tokenizer": "pte_optimized/tokenizer.json",
    },
    "Arm/whisper-small-int8-xnnpack-executorch": {
        "runtime": "executorch",
        "model": "pte_optimized/whisper_small_vivo_executorch_optimized.pte",
        "preprocessor": "pte_optimized/whisper_preprocessor.pte",
        "tokenizer": "pte_optimized/tokenizer.json",
    },
}

MODEL_SUFFIXES = {".pte", ".tflite"}
FILENAME_PATTERN = re.compile(r"(?m)^filename\s*:\s*([^\s#]+)")

LITERT_SUPPORT_FILES = (
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
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", compression=ZIP_STORED, allowZip64=True) as package:
        for path, archive_name in files:
            package.write(path, archive_name)


def repository_file(snapshot_dir: Path, filename: str) -> Path:
    repository_path = PurePosixPath(filename)
    if repository_path.is_absolute() or ".." in repository_path.parts:
        raise SystemExit(f"Expected a repository-relative filename: {filename}")
    return snapshot_dir.joinpath(*repository_path.parts)


def primary_model_filename(snapshot_dir: Path, requested: str | None) -> str:
    if requested:
        candidate = repository_file(snapshot_dir, requested)
        if not candidate.is_file():
            raise SystemExit(f"The requested model file was not found: {candidate}")
        if candidate.suffix.lower() not in MODEL_SUFFIXES:
            raise SystemExit("The primary model must be a .tflite or .pte file.")
        return PurePosixPath(requested).as_posix()

    for metadata_name in ("metadata.yaml", "metadata.yml"):
        metadata = snapshot_dir / metadata_name
        if not metadata.is_file():
            continue
        match = FILENAME_PATTERN.search(
            metadata.read_text(encoding="utf-8", errors="replace")
        )
        if match:
            filename = match.group(1).strip().strip(chr(39) + chr(34))
            candidate = repository_file(snapshot_dir, filename)
            if candidate.is_file() and candidate.suffix.lower() in MODEL_SUFFIXES:
                return PurePosixPath(filename).as_posix()

    candidates = [
        path
        for path in sorted(snapshot_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in MODEL_SUFFIXES
        and path.name != "whisper_preprocessor.pte"
        and ".cache" not in path.relative_to(snapshot_dir).parts
    ]
    if not candidates:
        raise SystemExit("The repository does not contain a .tflite or .pte model file.")
    if len(candidates) > 1:
        choices = ", ".join(
            path.relative_to(snapshot_dir).as_posix() for path in candidates
        )
        raise SystemExit(
            "The repository contains multiple model files. "
            f"Run the command again with --filename. Found: {choices}"
        )
    return candidates[0].relative_to(snapshot_dir).as_posix()


def required_support_file(
    snapshot_dir: Path,
    model_filename: str,
    support_name: str,
) -> Path:
    model_parent = PurePosixPath(model_filename).parent
    preferred_names = [str(model_parent / support_name), support_name]
    preferred = []
    for name in preferred_names:
        candidate = repository_file(snapshot_dir, name)
        if candidate.is_file() and candidate not in preferred:
            preferred.append(candidate)
    if preferred:
        return preferred[0]

    candidates = [
        path
        for path in sorted(snapshot_dir.rglob(support_name))
        if path.is_file() and ".cache" not in path.relative_to(snapshot_dir).parts
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"The repository does not contain {support_name}.")
    choices = ", ".join(
        path.relative_to(snapshot_dir).as_posix() for path in candidates
    )
    raise SystemExit(f"The repository contains multiple {support_name} files: {choices}")


def collect_compatible_package(
    snapshot_dir: Path,
    model_filename: str,
) -> list[tuple[Path, str]]:
    model = repository_file(snapshot_dir, model_filename)
    tokenizer = required_support_file(snapshot_dir, model_filename, "tokenizer.json")
    files = [(model, model.name), (tokenizer, "tokenizer.json")]

    if model.suffix.lower() == ".pte":
        preprocessor = required_support_file(
            snapshot_dir,
            model_filename,
            "whisper_preprocessor.pte",
        )
        files.insert(1, (preprocessor, "whisper_preprocessor.pte"))
        optional_names = PACKAGE_METADATA
    else:
        optional_names = tuple(
            name for name in LITERT_SUPPORT_FILES if name != "tokenizer.json"
        )

    model_parent = PurePosixPath(model_filename).parent
    included_names = {archive_name for _, archive_name in files}
    for name in optional_names:
        for relative_name in (str(model_parent / name), name):
            path = repository_file(snapshot_dir, relative_name)
            if path.is_file() and name not in included_names:
                files.append((path, name))
                included_names.add(name)
                break
    return files


def collect_litert_package(
    snapshot_dir: Path,
    model_filename: str,
) -> list[tuple[Path, str]]:
    model = repository_file(snapshot_dir, model_filename)
    tokenizer = repository_file(snapshot_dir, "tokenizer.json")
    if not model.is_file():
        raise SystemExit(f"Missing LiteRT model: {model}")
    if not tokenizer.is_file():
        raise SystemExit(f"Missing LiteRT tokenizer: {tokenizer}")

    files = [(model, model.name), (tokenizer, "tokenizer.json")]
    for name in LITERT_SUPPORT_FILES:
        if name == "tokenizer.json":
            continue
        path = repository_file(snapshot_dir, name)
        if path.is_file():
            files.append((path, name))
    return files


def executorch_support_files(
    model_filename: str,
    model: dict[str, str],
) -> tuple[str, str]:
    if model_filename == model["model"]:
        return model["preprocessor"], model["tokenizer"]

    model_directory = PurePosixPath(model_filename).parent
    return (
        str(model_directory / "whisper_preprocessor.pte"),
        str(model_directory / "tokenizer.json"),
    )


def collect_executorch_package(
    snapshot_dir: Path,
    model_filename: str,
    preprocessor_filename: str,
    tokenizer_filename: str,
) -> list[tuple[Path, str]]:
    model = repository_file(snapshot_dir, model_filename)
    preprocessor = repository_file(snapshot_dir, preprocessor_filename)
    tokenizer = repository_file(snapshot_dir, tokenizer_filename)
    if not model.is_file():
        raise SystemExit(f"Missing ExecuTorch model: {model}")
    if not preprocessor.is_file():
        raise SystemExit(f"Missing ExecuTorch preprocessor: {preprocessor}")
    if not tokenizer.is_file():
        raise SystemExit(f"Missing ExecuTorch tokenizer: {tokenizer}")

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
    parser.add_argument(
        "--filename",
        help="Repository-relative primary model filename when selecting an alternative",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Download/package directory (default: models)",
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print only the generated Android package path to standard output",
    )
    args = parser.parse_args()

    model = VERIFIED_MODELS.get(args.repo_id)
    output_dir = args.output_dir
    snapshot_dir = output_dir / args.repo_id.replace("/", "__")
    output_stream = sys.stderr if args.print_path else sys.stdout
    if model is None:
        print(f"Downloading {args.repo_id} to {snapshot_dir} ...", file=output_stream)
        snapshot_download(repo_id=args.repo_id, local_dir=snapshot_dir)
        filename = primary_model_filename(snapshot_dir, args.filename)
        files = collect_compatible_package(snapshot_dir, filename)
    else:
        filename = args.filename or model["model"]
        expected_suffix = ".tflite" if model["runtime"] == "litert" else ".pte"
        if PurePosixPath(filename).suffix.lower() != expected_suffix:
            raise SystemExit(
                f"Expected a {expected_suffix} model filename for {args.repo_id}: {filename}"
            )
        print(
            f"Downloading {args.repo_id}/{filename} to {snapshot_dir} ...",
            file=output_stream,
        )

    if model is not None and model["runtime"] == "litert":
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=snapshot_dir,
            allow_patterns=[filename, *LITERT_SUPPORT_FILES],
        )
        files = collect_litert_package(snapshot_dir, filename)
    elif model is not None:
        preprocessor_filename, tokenizer_filename = executorch_support_files(
            filename,
            model,
        )
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=snapshot_dir,
            allow_patterns=[
                filename,
                preprocessor_filename,
                tokenizer_filename,
                *PACKAGE_METADATA,
            ],
        )
        files = collect_executorch_package(
            snapshot_dir,
            filename,
            preprocessor_filename,
            tokenizer_filename,
        )

    archive = output_dir / f"{args.repo_id.split('/')[-1]}.zip"
    package_files(archive, files)

    resolved_archive = archive.resolve()
    if args.print_path:
        print(resolved_archive)
    else:
        print(f"Model directory: {snapshot_dir.resolve()}")
        print(f"Android package: {resolved_archive}")


if __name__ == "__main__":
    main()
