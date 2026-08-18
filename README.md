# Whisper Journal

Whisper Journal is a private, on-device speech-to-text sample for Arm64 Android devices. Record up to 30 seconds of speech, transcribe it locally, then copy, save, clear, or replace the journal entry.

## Application views

<p align="center">
  <img src="images/whisper-startup.png" width="31%" alt="Whisper Journal before the model archive has been imported">
  <img src="images/whisper-model-ready.png" width="31%" alt="Whisper Journal with a speech model ready to record">
  <img src="images/whisper-generated.png" width="31%" alt="Whisper Journal displaying a transcript generated locally on Android">
</p>

## Registered model packages

| Model | Runtime adapter | Hugging Face repository ID | Android package |
| --- | --- | --- | --- |
| Whisper Tiny INT8 | ExecuTorch | [`Arm/whisper-tiny-int8-xnnpack-executorch`](https://huggingface.co/Arm/whisper-tiny-int8-xnnpack-executorch) | `whisper-tiny-int8-xnnpack-executorch.zip` |
| Whisper Small INT8 | ExecuTorch | [`Arm/whisper-small-int8-xnnpack-executorch`](https://huggingface.co/Arm/whisper-small-int8-xnnpack-executorch) | `whisper-small-int8-xnnpack-executorch.zip` |
| Whisper Base INT8 | LiteRT | [`Arm/whisper-base-int8-litert`](https://huggingface.co/Arm/whisper-base-int8-litert) | `whisper-base-int8-litert.zip` |
| Whisper Medium INT8 | LiteRT | [`Arm/whisper-medium-int8-litert`](https://huggingface.co/Arm/whisper-medium-int8-litert) | `whisper-medium-int8-litert.zip` |
| Whisper Large V3 INT8 | LiteRT | [`Arm/whisper-large-v3-int8-litert`](https://huggingface.co/Arm/whisper-large-v3-int8-litert) | `whisper-large-v3-int8-litert.zip` |

## Build the application

Open this directory in Android Studio, allow Gradle to synchronize, connect an Arm64 device running Android 9 (API 28) or later, and run the `app` configuration.

The project uses Java 17, ExecuTorch 1.3.1, and LiteRT 2.1.6.

## Package a model from Hugging Face

Install the Hugging Face Hub client and run the included downloader with any repository ID from the table:

```console
python -m pip install huggingface_hub
python download_model.py --repo-id Arm/whisper-tiny-int8-xnnpack-executorch
```

For ExecuTorch, the script downloads the INT8 model and audio preprocessor and adds `tokenizer.json` from the matching OpenAI Whisper repository. For LiteRT, it downloads the INT8 model and its tokenizer and configuration files. It creates the Android ZIP under `models/`; copy that ZIP to the device.

## Transcribe a journal entry

1. Expand **AI model** and select the matching model.
2. Tap **Add or change model** and choose its ZIP file.
3. Tap **Record**, speak for up to 30 seconds, and tap **Stop recording**.
4. Copy, save, clear, or record the entry again.

Audio and transcripts remain on the device. The current adapters perform English transcription and keep only one model runtime loaded at a time.

## Application structure

- `MainActivity.java` owns the common journaling interface and background work.
- `ModelRegistry.java` describes the five registered model packages.
- `AdapterRegistry.java` maps each descriptor to the ExecuTorch or LiteRT adapter.
- `ExecuTorchWhisperAdapter.java` runs the Tiny and Small ExecuTorch packages.
- `litert/LiteRtWhisperAdapter.java` provides profiles for the Base, Medium, and Large V3 LiteRT exports.
- `ModelPackageImporter.java` validates and installs selected model ZIPs in private app storage.
