package org.arm.learningpath.whisper;

@FunctionalInterface
public interface ProgressListener {
    ProgressListener NONE = (message, fraction) -> { };

    void onProgress(String message, float fraction);
}
