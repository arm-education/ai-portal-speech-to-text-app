package org.arm.learningpath.speech;

@FunctionalInterface
public interface ProgressListener {
    ProgressListener NONE = (message, fraction) -> { };

    void onProgress(String message, float fraction);
}
