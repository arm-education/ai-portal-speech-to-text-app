package org.arm.learningpath.whisper;

import java.util.Objects;

public final class WhisperModelDescriptor {
    private final String id;
    private final String displayName;
    private final String adapterId;
    private final String sourceId;
    private final String packageHint;
    private final int melBins;
    private final int vocabularySize;
    private final String configurationId;

    public WhisperModelDescriptor(
            String id,
            String displayName,
            String adapterId,
            String sourceId,
            String packageHint,
            int melBins,
            int vocabularySize
    ) {
        this(
                id,
                displayName,
                adapterId,
                sourceId,
                packageHint,
                melBins,
                vocabularySize,
                id
        );
    }

    public WhisperModelDescriptor(
            String id,
            String displayName,
            String adapterId,
            String sourceId,
            String packageHint,
            int melBins,
            int vocabularySize,
            String configurationId
    ) {
        this.id = requireText(id, "id");
        this.displayName = requireText(displayName, "displayName");
        this.adapterId = requireText(adapterId, "adapterId");
        this.sourceId = sourceId == null ? "" : sourceId;
        this.packageHint = requireText(packageHint, "packageHint");
        if (melBins <= 0 || vocabularySize <= 0) {
            throw new IllegalArgumentException("Model dimensions must be positive");
        }
        this.melBins = melBins;
        this.vocabularySize = vocabularySize;
        this.configurationId = requireText(configurationId, "configurationId");
    }

    public String id() {
        return id;
    }

    public String displayName() {
        return displayName;
    }

    public String adapterId() {
        return adapterId;
    }

    /** Hugging Face repository ID when the package is distributed there. */
    public String sourceId() {
        return sourceId;
    }

    public String packageHint() {
        return packageHint;
    }

    public int melBins() {
        return melBins;
    }

    public int vocabularySize() {
        return vocabularySize;
    }

    /** Supplied adapter profile that defines the complete execution contract. */
    public String configurationId() {
        return configurationId;
    }

    @Override
    public String toString() {
        return displayName;
    }

    @Override
    public boolean equals(Object value) {
        return value instanceof WhisperModelDescriptor
                && id.equals(((WhisperModelDescriptor) value).id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    private static String requireText(String value, String fieldName) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(fieldName + " must not be empty");
        }
        return value;
    }
}
