package org.arm.learningpath.whisper;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class AdapterRegistry implements AutoCloseable {
    public static final String EXECUTORCH_ID = "executorch";
    public static final String LITERT_ID = "litert";

    private final Map<String, SpeechToTextAdapter> adapters;

    public AdapterRegistry(
            SpeechToTextAdapter execuTorchAdapter,
            SpeechToTextAdapter liteRtAdapter
    ) {
        if (execuTorchAdapter == null || !EXECUTORCH_ID.equals(execuTorchAdapter.id())) {
            throw new IllegalArgumentException("Expected the ExecuTorch adapter");
        }
        if (liteRtAdapter == null || !LITERT_ID.equals(liteRtAdapter.id())) {
            throw new IllegalArgumentException("Expected the LiteRT adapter");
        }
        Map<String, SpeechToTextAdapter> registered = new LinkedHashMap<>();
        registered.put(EXECUTORCH_ID, execuTorchAdapter);
        registered.put(LITERT_ID, liteRtAdapter);
        adapters = Collections.unmodifiableMap(registered);
    }

    public SpeechToTextAdapter forModel(WhisperModelDescriptor descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException("Select a model first");
        }
        SpeechToTextAdapter adapter = adapters.get(descriptor.adapterId());
        if (adapter == null) {
            throw new IllegalArgumentException(
                    "No adapter is registered for " + descriptor.adapterId()
            );
        }
        return adapter;
    }

    @Override
    public void close() {
        for (SpeechToTextAdapter adapter : adapters.values()) {
            adapter.close();
        }
    }
}
