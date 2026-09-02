#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


PLACEHOLDER_PATTERN = re.compile(
    r"<(?:MODEL|RUNTIME|ADAPTER|PLACEHOLDER|TODO)[A-Z0-9_-]*>"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check files produced for a generated Android speech adapter."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--layout", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    source_directory = project_root / (
        "app/src/main/java/org/arm/learningpath/whisper"
    )
    registry = source_directory / "GeneratedAdapterRegistry.java"
    dependencies = project_root / "app/generated-runtime-dependencies.gradle.kts"
    adapter = args.adapter if args.adapter.is_absolute() else project_root / args.adapter
    paths = [adapter, registry, dependencies]
    if args.layout:
        layout = args.layout if args.layout.is_absolute() else project_root / args.layout
        paths.append(layout)

    errors = []
    for path in paths:
        if not path.is_file():
            errors.append(f"Missing file: {path}")
            continue
        content = path.read_text(encoding="utf-8")
        if PLACEHOLDER_PATTERN.search(content):
            errors.append(f"Unresolved placeholder in {path}")

    if adapter.is_file():
        adapter_source = adapter.read_text(encoding="utf-8")
        if "implements SpeechToTextAdapter" not in adapter_source:
            errors.append("The generated class must implement SpeechToTextAdapter")

    if registry.is_file() and "return List.of();" in registry.read_text(encoding="utf-8"):
        errors.append("GeneratedAdapterRegistry does not register an adapter or model")

    if errors:
        print("Generated adapter validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Generated adapter file validation passed")
    print("Run the Gradle build, lint, and on-device transcription test next")


if __name__ == "__main__":
    main()
