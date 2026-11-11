package org.arm.learningpath.speech;

import java.util.List;

/** Learning Path extension point for models that reuse an existing adapter. */
final class CompatibleModelRegistry {
    private CompatibleModelRegistry() {
    }

    static List<SpeechModelDescriptor> models() {
        // Add compatible model descriptors here.
        return List.of();
    }
}
