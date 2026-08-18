package org.arm.learningpath.whisper;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class ModelRegistry {
    private static final List<WhisperModelDescriptor> MODELS = Collections.unmodifiableList(
            Arrays.asList(
                    new WhisperModelDescriptor(
                            "whisper-tiny-executorch-v2",
                            "Whisper Tiny · ExecuTorch V2",
                            AdapterRegistry.EXECUTORCH_ID,
                            "",
                            "whisper_tiny_executorch_v2.zip",
                            80,
                            51865
                    ),
                    new WhisperModelDescriptor(
                            "whisper-small-executorch-v2",
                            "Whisper Small · ExecuTorch V2",
                            AdapterRegistry.EXECUTORCH_ID,
                            "",
                            "whisper_small_executorch_v2.zip",
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
