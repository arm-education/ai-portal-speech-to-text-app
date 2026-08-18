package org.arm.learningpath.whisper;

public final class TranscriptionResult {
    private final String text;
    private final long elapsedMillis;
    private final int tokenCount;
    private final String backend;

    public TranscriptionResult(
            String text,
            long elapsedMillis,
            int tokenCount,
            String backend
    ) {
        this.text = text == null ? "" : text;
        this.elapsedMillis = elapsedMillis;
        this.tokenCount = tokenCount;
        this.backend = backend == null ? "" : backend;
    }

    public String text() {
        return text;
    }

    public long elapsedMillis() {
        return elapsedMillis;
    }

    public int tokenCount() {
        return tokenCount;
    }

    public String backend() {
        return backend;
    }
}
