# Whisper Journal Android application

This example application accompanies the [Arm Learning Path for running speech recognition models from the Arm AI Portal](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/ai-portal-audio-to-text). It is intended for learning how models run on devices and is not a reference production application. It is provided under the [Arm Education End User License Agreement](LICENSE.md).

This Android application records speech and transcribes it locally on an Arm64 phone or emulator. It includes two supplied adapters:

- `ExecuTorchWhisperAdapter` runs the Whisper Tiny and Small INT8 packages with ExecuTorch and XNNPACK.
- `LiteRtWhisperAdapter` provides execution profiles for the Whisper Base, Medium, and Large V3 INT8 packages with LiteRT and XNNPACK.

The application imports model packages at run time, so the model files are not stored in the Android application package (APK).

## Application views

<p align="center">
  <img src="images/whisper-startup.png" width="31%" alt="Whisper Journal before a model package has been imported">
  <img src="images/whisper-model-ready.png" width="31%" alt="Whisper Journal with a speech recognition model ready to record">
  <img src="images/whisper-generated.png" width="31%" alt="Whisper Journal displaying a transcript generated locally on Android">
</p>

Whisper Journal records up to 30 seconds of mono audio and performs English transcription on the device. The recording and transcript remain local.

## Requirements

- Android Studio with Android SDK 35
- Java 17, supplied by Android Studio
- An Arm64 Android device running Android 9, API 28, or later
- Python 3 and the Hugging Face Hub package for the model downloader
- One registered model package downloaded from the Arm AI Portal

## Registered model packages

The model registry selects one of the two supplied adapters for the chosen package. The importer checks and installs the package, and the adapter handles audio preprocessing, runtime calls, and token decoding.

| Model | Runtime | Import this package | Validation |
| --- | --- | --- | --- |
| [Whisper Tiny INT8](https://huggingface.co/Arm/whisper-tiny-int8-xnnpack-executorch) | ExecuTorch | `whisper-tiny-int8-xnnpack-executorch.zip` | Validated on an Arm64 Android phone |
| [Whisper Small INT8](https://huggingface.co/Arm/whisper-small-int8-xnnpack-executorch) | ExecuTorch | `whisper-small-int8-xnnpack-executorch.zip` | Validated on an Arm64 Android phone |
| [Whisper Base INT8](https://huggingface.co/Arm/whisper-base-int8-litert) | LiteRT | `whisper-base-int8-litert.zip` | Validated on an Arm64 Android phone |
| [Whisper Medium INT8](https://huggingface.co/Arm/whisper-medium-int8-litert) | LiteRT | `whisper-medium-int8-litert.zip` | Registered; target-device verification pending |
| [Whisper Large V3 INT8](https://huggingface.co/Arm/whisper-large-v3-int8-litert) | LiteRT | `whisper-large-v3-int8-litert.zip` | Registered; target-device verification pending |

## Download a model

Create a Python virtual environment and install the Hugging Face Hub package.

On macOS or Linux:

```bash
python3 -m venv .hf-venv
source .hf-venv/bin/activate
python -m pip install --upgrade huggingface_hub
```

On Windows PowerShell:

```powershell
py -m venv .hf-venv
.\.hf-venv\Scripts\Activate.ps1
python -m pip install --upgrade huggingface_hub
```

Set the repository ID for one of the registered models, then run the included download script. The example below uses Whisper Base LiteRT.

On macOS or Linux:

```bash
MODEL_ID="Arm/whisper-base-int8-litert"
MODEL_PACKAGE="models/whisper-base-int8-litert.zip"

python download_model.py --repo-id "$MODEL_ID"
printf 'Model package: %s\n' "$MODEL_PACKAGE"
```

On Windows PowerShell:

```powershell
$MODEL_ID = "Arm/whisper-base-int8-litert"
$MODEL_PACKAGE = "models\whisper-base-int8-litert.zip"

python download_model.py --repo-id $MODEL_ID
Write-Output "Model package: $MODEL_PACKAGE"
```

For an ExecuTorch model, the script downloads the INT8 model and audio preprocessor, adds `tokenizer.json` from the matching OpenAI Whisper repository, and creates one ZIP package. For a LiteRT model, it downloads the INT8 model, tokenizer, and configuration files and creates the ZIP package.

Copy the package to the Android **Downloads** directory through ADB. Run this command in the same terminal session:

```console
adb push "$MODEL_PACKAGE" /sdcard/Download/
```

## Open and run the application

1. Clone or download this repository.
2. Open the repository root in Android Studio.
3. Wait for Gradle sync to finish.
4. Connect an Arm64 Android phone or start an Arm64 emulator.
5. Select the `app` configuration and run it.
6. Expand **AI model**, then select the model that matches the downloaded package.
7. Select **Add or change model** and choose the ZIP package.
8. Select **Record**, speak for up to 30 seconds, then select **Stop recording**.
9. Copy or save the transcript. Use **More** to clear the entry.

The application stores the imported model in its private files directory. Clearing application data or uninstalling the application removes imported models.

## Register another compatible model

A model that matches an existing adapter's package, callable methods, tensor shapes, preprocessing, tokenizer, and decoder contract can reuse that adapter after you add a `WhisperModelDescriptor` to `ModelRegistry.java`.

For example, a LiteRT Whisper package can be described with:

```java
new WhisperModelDescriptor(
        "my-whisper-litert",
        "My Whisper model - LiteRT",
        AdapterRegistry.LITERT_ID,
        "Arm/<model-repository-name>",
        "my-whisper-litert.zip",
        80,
        51865
)
```

Add the descriptor to the list returned by `ModelRegistry.models()`. Also add the model's dimensions and decoder constants to the matching adapter profile, and update the package identity checks in `ModelPackageImporter.java`. Rebuild the APK and test the package on an Arm64 Android device.

This route reuses an existing adapter. It is appropriate only when the complete package and inference contract match that adapter.

## Extend the application

The application resolves model descriptors through `AdapterRegistry.java`. The supplied adapters support the split ExecuTorch Whisper export and the registered LiteRT `encode` and `decode` profiles.

Implement `SpeechToTextAdapter` when a model changes the package layout, callable methods, tensor contract, runtime, preprocessing, or token decoding. Register the adapter in `AdapterRegistry.java`, construct it in `MainActivity.java`, and rebuild the APK. Packages that require additional model files also need corresponding changes to `ModelPackageImporter.java`.

## License

This project is provided under the [Arm Education End User License Agreement](LICENSE.md).
