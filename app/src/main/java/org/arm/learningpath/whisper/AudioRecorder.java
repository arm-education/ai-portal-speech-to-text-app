package org.arm.learningpath.whisper;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import java.io.IOException;
import java.util.Arrays;

/** Start/stop microphone capture for the 16 kHz waveform expected by Whisper. */
public final class AudioRecorder {
    public static final int SAMPLE_RATE = 16000;
    public static final int MAX_DURATION_SECONDS = 30;
    private static final int MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION_SECONDS;

    private final Object lock = new Object();
    private AudioRecord recorder;
    private Thread captureThread;
    private boolean recording;
    private boolean stopRequested;
    private float[] completedAudio;
    private IOException captureFailure;
    private MaxDurationListener maximumDurationListener;

    public AudioRecorder() {
    }

    public void start(MaxDurationListener listener) throws IOException {
        synchronized (lock) {
            if (recording || captureThread != null) {
                throw new IllegalStateException("A recording is already in progress");
            }

            int minimumBuffer = AudioRecord.getMinBufferSize(
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
            );
            if (minimumBuffer <= 0) {
                throw new IOException("Android could not create a 16 kHz audio buffer");
            }

            try {
                recorder = new AudioRecord(
                        MediaRecorder.AudioSource.VOICE_RECOGNITION,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        Math.max(minimumBuffer, SAMPLE_RATE * 2)
                );
            } catch (SecurityException error) {
                throw new IOException("Microphone permission has not been granted", error);
            }
            if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                recorder.release();
                recorder = null;
                throw new IOException("Android could not initialize the microphone");
            }

            stopRequested = false;
            completedAudio = null;
            captureFailure = null;
            maximumDurationListener = listener;
            try {
                recorder.startRecording();
            } catch (SecurityException | IllegalStateException error) {
                recorder.release();
                recorder = null;
                throw new IOException("Android could not start the microphone", error);
            }
            recording = true;
            captureThread = new Thread(this::captureLoop, "whisper-audio-capture");
            captureThread.start();
        }
    }

    public float[] stop() throws IOException {
        AudioRecord activeRecorder;
        Thread activeThread;
        synchronized (lock) {
            activeRecorder = recorder;
            activeThread = captureThread;
            if (activeThread == null) {
                if (captureFailure != null) {
                    throw captureFailure;
                }
                return completedAudio == null ? new float[0] : completedAudio.clone();
            }
            stopRequested = true;
        }

        if (activeRecorder != null) {
            try {
                activeRecorder.stop();
            } catch (IllegalStateException ignored) {
                // The capture thread may already have reached the 30-second limit.
            }
        }
        if (activeThread != Thread.currentThread()) {
            try {
                activeThread.join(2_000L);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while stopping the recording", error);
            }
        }

        synchronized (lock) {
            if (captureThread != null) {
                throw new IOException("The microphone did not stop cleanly");
            }
            if (captureFailure != null) {
                throw captureFailure;
            }
            return completedAudio == null ? new float[0] : completedAudio.clone();
        }
    }

    public boolean isRecording() {
        synchronized (lock) {
            return recording;
        }
    }

    public void release() {
        try {
            stop();
        } catch (IOException ignored) {
            // Nothing else can be recovered while the Activity is closing.
        }
    }

    public static float peak(float[] audio) {
        float peak = 0.0f;
        if (audio != null) {
            for (float sample : audio) {
                peak = Math.max(peak, Math.abs(sample));
            }
        }
        return peak;
    }

    public static float rms(float[] audio) {
        if (audio == null || audio.length == 0) {
            return 0.0f;
        }
        double sumSquares = 0.0;
        for (float sample : audio) {
            sumSquares += sample * sample;
        }
        return (float) Math.sqrt(sumSquares / audio.length);
    }

    private void captureLoop() {
        AudioRecord activeRecorder;
        MaxDurationListener listener;
        synchronized (lock) {
            activeRecorder = recorder;
            listener = maximumDurationListener;
        }

        short[] pcm = new short[MAX_SAMPLES];
        short[] buffer = new short[4096];
        int sampleCount = 0;
        boolean reachedMaximum = false;
        IOException failure = null;
        try {
            while (sampleCount < MAX_SAMPLES) {
                synchronized (lock) {
                    if (stopRequested) {
                        break;
                    }
                }
                int requested = Math.min(buffer.length, MAX_SAMPLES - sampleCount);
                int read = activeRecorder.read(
                        buffer,
                        0,
                        requested,
                        AudioRecord.READ_BLOCKING
                );
                if (read < 0) {
                    synchronized (lock) {
                        if (stopRequested) {
                            break;
                        }
                    }
                    throw new IOException("AudioRecord failed with code " + read);
                }
                if (read == 0) {
                    continue;
                }
                System.arraycopy(buffer, 0, pcm, sampleCount, read);
                sampleCount += read;
            }
            reachedMaximum = sampleCount >= MAX_SAMPLES;
        } catch (IOException error) {
            failure = error;
        } catch (RuntimeException error) {
            synchronized (lock) {
                if (!stopRequested) {
                    failure = new IOException("Microphone capture failed", error);
                }
            }
        } finally {
            try {
                if (activeRecorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    activeRecorder.stop();
                }
            } catch (IllegalStateException ignored) {
                // stop() may already have been called by the UI thread.
            }
            activeRecorder.release();
        }

        float[] audio = new float[sampleCount];
        for (int index = 0; index < sampleCount; index++) {
            audio[index] = pcm[index] / 32768.0f;
        }
        // Drop the large temporary buffer before the UI starts model inference.
        Arrays.fill(pcm, (short) 0);

        synchronized (lock) {
            completedAudio = audio;
            captureFailure = failure;
            recording = false;
            recorder = null;
            captureThread = null;
            maximumDurationListener = null;
            lock.notifyAll();
        }
        if (reachedMaximum && failure == null && listener != null) {
            listener.onMaximumDurationReached(audio.clone());
        }
    }

    @FunctionalInterface
    public interface MaxDurationListener {
        void onMaximumDurationReached(float[] audio16Khz);
    }
}
