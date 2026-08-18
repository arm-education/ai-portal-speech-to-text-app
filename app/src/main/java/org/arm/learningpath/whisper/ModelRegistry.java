package org.arm.learningpath.whisper;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class ModelRegistry {
    private static final List<WhisperModelDescriptor> MODELS = Collections.unmodifiableList(
            Arrays.asList(
                    new WhisperModelDescriptor(
                            "whisper-tiny-int8-xnnpack-executorch",
                            "Whisper Tiny · ExecuTorch",
                            AdapterRegistry.EXECUTORCH_ID,
                            "Arm/whisper-tiny-int8-xnnpack-executorch",
                            "whisper-tiny-int8-xnnpack-executorch.zip",
                            80,
                            51865
                    ),
                    new WhisperModelDescriptor(
                            "whisper-small-int8-xnnpack-executorch",
                            "Whisper Small · ExecuTorch",
                            AdapterRegistry.EXECUTORCH_ID,
                            "Arm/whisper-small-int8-xnnpack-executorch",
                            "whisper-small-int8-xnnpack-executorch.zip",
                            80,
                            51865
                    ),
                    new WhisperModelDescriptor(
                            "whisper-base-litert",
                            "Whisper Base · LiteRT",
                            AdapterRegistry.LITERT_ID,
                            "Arm/whisper-base-int8-litert",
                            "whisper-base-int8-litert.zip",
                            80,
                            51865
                    ),
                    new WhisperModelDescriptor(
                            "whisper-medium-litert",
                            "Whisper Medium · LiteRT",
                            AdapterRegistry.LITERT_ID,
                            "Arm/whisper-medium-int8-litert",
                            "whisper-medium-int8-litert.zip",
                            80,
                            51865
                    ),
                    new WhisperModelDescriptor(
                            "whisper-large-v3-litert",
                            "Whisper Large V3 · LiteRT",
                            AdapterRegistry.LITERT_ID,
                            "Arm/whisper-large-v3-int8-litert",
                            "whisper-large-v3-int8-litert.zip",
                            128,
                            51866
                    )
            )
    );

    private ModelRegistry() {
    }

    public static List<WhisperModelDescriptor> models() {
        return MODELS;
    }

    public static WhisperModelDescriptor findById(String modelId) {
        for (WhisperModelDescriptor model : MODELS) {
            if (model.id().equals(modelId)) {
                return model;
            }
        }
        throw new IllegalArgumentException("Unknown Whisper model: " + modelId);
    }
}
