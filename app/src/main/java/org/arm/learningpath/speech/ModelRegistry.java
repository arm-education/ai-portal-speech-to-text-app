package org.arm.learningpath.speech;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class ModelRegistry {
    // Supplied models; use the extension registries for the Learning Path exercises.
    private static final List<SpeechModelDescriptor> BUILT_IN_MODELS = Collections.unmodifiableList(
            Arrays.asList(
                    new SpeechModelDescriptor(
                            "whisper-tiny-int8-xnnpack-executorch",
                            "Whisper Tiny · ExecuTorch",
                            AdapterRegistry.EXECUTORCH_ID,
                            "Arm/whisper-tiny-int8-xnnpack-executorch",
                            "whisper-tiny-int8-xnnpack-executorch.zip",
                            80,
                            51865
                    ),
                    new SpeechModelDescriptor(
                            "whisper-small-int8-xnnpack-executorch",
                            "Whisper Small · ExecuTorch",
                            AdapterRegistry.EXECUTORCH_ID,
                            "Arm/whisper-small-int8-xnnpack-executorch",
                            "whisper-small-int8-xnnpack-executorch.zip",
                            80,
                            51865
                    ),
                    new SpeechModelDescriptor(
                            "whisper-base-int8-xnnpack-executorch-vivo-x300",
                            "Whisper Base · ExecuTorch",
                            AdapterRegistry.EXECUTORCH_SPLIT_ID,
                            "Arm/whisper-base-int8-xnnpack-executorch-vivo-x300",
                            "whisper-base-int8-xnnpack-executorch-vivo-x300.zip",
                            80,
                            51865
                    ),
                    new SpeechModelDescriptor(
                            "whisper-base-litert",
                            "Whisper Base · LiteRT",
                            AdapterRegistry.LITERT_ID,
                            "Arm/whisper-base-int8-litert",
                            "whisper-base-int8-litert.zip",
                            80,
                            51865
                    ),
                    new SpeechModelDescriptor(
                            "whisper-medium-litert",
                            "Whisper Medium · LiteRT",
                            AdapterRegistry.LITERT_ID,
                            "Arm/whisper-medium-int8-litert",
                            "whisper-medium-int8-litert.zip",
                            80,
                            51865
                    ),
                    new SpeechModelDescriptor(
                            "whisper-large-v3-litert",
                            "Whisper Large V3 · LiteRT",
                            AdapterRegistry.LITERT_ID,
                            "Arm/whisper-large-v3-int8-litert",
                            "whisper-large-v3-int8-litert.zip",
                            128,
                            51866
                    ),
                    new SpeechModelDescriptor(
                            "conformer-ctc-int8-xnnpack-executorch-vivo-x300",
                            "Conformer CTC · ExecuTorch",
                            AdapterRegistry.EXECUTORCH_CTC_ID,
                            "Arm/conformer-ctc-int8-xnnpack-executorch-vivo-x300",
                            "conformer-ctc-int8-xnnpack-executorch-vivo-x300.zip",
                            0,
                            32,
                            ExecuTorchCtcAdapter.CONFIG_CONFORMER_LARGE
                    )
            )
    );
    private static final List<SpeechModelDescriptor> MODELS = createModels();

    private ModelRegistry() {
    }

    private static List<SpeechModelDescriptor> createModels() {
        List<SpeechModelDescriptor> models = new ArrayList<>(BUILT_IN_MODELS);
        models.addAll(CompatibleModelRegistry.models());
        models.addAll(GeneratedAdapterRegistry.models());
        return Collections.unmodifiableList(models);
    }

    public static List<SpeechModelDescriptor> models() {
        return MODELS;
    }

    public static SpeechModelDescriptor findById(String modelId) {
        for (SpeechModelDescriptor model : MODELS) {
            if (model.id().equals(modelId)) {
                return model;
            }
        }
        throw new IllegalArgumentException("Unknown speech model: " + modelId);
    }
}
