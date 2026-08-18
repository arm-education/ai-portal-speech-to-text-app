package org.arm.learningpath.whisper;

import java.io.File;

/** Runtime boundary shared by the app's ExecuTorch and LiteRT implementations. */
public interface SpeechToTextAdapter extends AutoCloseable {
    String id();

    /**
     * Loads the selected package. Calling this again replaces the currently loaded model.
     */
    void load(WhisperModelDescriptor descriptor, File modelDirectory) throws Exception;

    boolean isLoaded();

    TranscriptionResult transcribe(
            float[] audio16Khz,
            ProgressListener progressListener
    ) throws Exception;

    /** Releases native state. The adapter may be loaded again afterwards. */
    @Override
    void close();
}
