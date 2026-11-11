package org.arm.learningpath.speech;

import java.util.List;

/** Learning Path extension point for new adapters and their model descriptors. */
final class GeneratedAdapterRegistry {
    private GeneratedAdapterRegistry() {
    }

    static List<SpeechToTextAdapter> adapters() {
        // Add generated adapter instances here.
        return List.of();
    }

    static List<SpeechModelDescriptor> models() {
        // Add the descriptors for those adapters here.
        return List.of();
    }
}
