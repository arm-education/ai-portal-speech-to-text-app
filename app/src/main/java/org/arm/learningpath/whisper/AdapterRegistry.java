package org.arm.learningpath.whisper;

import org.arm.learningpath.whisper.litert.LiteRtWhisperAdapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class AdapterRegistry implements AutoCloseable {
    public static final String EXECUTORCH_ID = "executorch";
    public static final String LITERT_ID = "litert";

    private final Map<String, SpeechToTextAdapter> adapters;

    public AdapterRegistry() {
        List<SpeechToTextAdapter> adapters = new ArrayList<>();
        adapters.add(new ExecuTorchWhisperAdapter());
        adapters.add(new LiteRtWhisperAdapter());
        adapters.addAll(GeneratedAdapterRegistry.adapters());

        Map<String, SpeechToTextAdapter> registered = new LinkedHashMap<>();
        for (SpeechToTextAdapter adapter : adapters) {
            if (adapter == null || adapter.id() == null || adapter.id().trim().isEmpty()) {
                throw new IllegalArgumentException("Every speech adapter needs an ID");
            }
            if (registered.put(adapter.id(), adapter) != null) {
                throw new IllegalArgumentException("Duplicate speech adapter ID: " + adapter.id());
            }
        }
        this.adapters = Collections.unmodifiableMap(registered);
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
