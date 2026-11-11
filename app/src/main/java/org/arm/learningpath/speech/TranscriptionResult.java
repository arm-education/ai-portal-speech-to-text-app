package org.arm.learningpath.speech;

public final class TranscriptionResult {
    private final String text;
    private final long elapsedMillis;
    private final String backend;

    public TranscriptionResult(
            String text,
            long elapsedMillis,
            String backend
    ) {
        this.text = text == null ? "" : text;
        this.elapsedMillis = elapsedMillis;
        this.backend = backend == null ? "" : backend;
    }

    public String text() {
        return text;
    }

    public long elapsedMillis() {
        return elapsedMillis;
    }

    public String backend() {
        return backend;
    }
}
