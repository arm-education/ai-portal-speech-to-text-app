package org.arm.learningpath.whisper;

import android.util.Log;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Adapter for the optimized Tiny and Small V2 ExecuTorch packages. */
public final class ExecuTorchWhisperAdapter implements SpeechToTextAdapter {
    private static final String TAG = "WhisperExecuTorch";
    private static final int DECODER_START_TOKEN_ID = 50258;
    private static final int EOS_TOKEN_ID = 50257;
    private static final int[] FORCED_PREFIX_IDS = {50259, 50359, 50363};
    private static final int MAX_GENERATION_TOKENS = 128;
    private static final int REPETITION_GUARD_REPEATS = 3;
    private static final int REPETITION_GUARD_MIN_PATTERN = 2;
    private static final int REPETITION_GUARD_MAX_PATTERN = 16;
    private static final int[] SUPPRESS_TOKENS = {
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
            63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220,
            1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488,
            3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959,
            10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306,
            16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109,
            30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50359,
            50360, 50361
    };
    private static final Set<Integer> SUPPRESSED = createSuppressedSet();

    private Module model;
    private Module preprocessor;
    private WhisperTokenizer tokenizer;
    private String backend = "";

    public ExecuTorchWhisperAdapter() {
    }

    @Override
    public String id() {
        return AdapterRegistry.EXECUTORCH_ID;
    }

    @Override
    public synchronized void load(
            WhisperModelDescriptor descriptor,
            File modelDirectory
    ) throws Exception {
        if (descriptor == null || !id().equals(descriptor.adapterId())) {
            throw new IllegalArgumentException("This package is not an ExecuTorch V2 model");
        }
        File modelFile = requiredFile(modelDirectory, "model.pte");
        File preprocessorFile = requiredFile(modelDirectory, "whisper_preprocessor.pte");
        File tokenizerFile = requiredFile(modelDirectory, "tokenizer.json");

        close();
        Module nextModel = null;
        Module nextPreprocessor = null;
        try {
            nextModel = Module.load(modelFile.getAbsolutePath(), Module.LOAD_MODE_MMAP, 4);
            nextPreprocessor = Module.load(
                    preprocessorFile.getAbsolutePath(),
                    Module.LOAD_MODE_MMAP,
                    2
            );
            WhisperTokenizer nextTokenizer = new WhisperTokenizer(tokenizerFile);

            Set<String> methods = new HashSet<>(Arrays.asList(nextModel.getMethods()));
            if (!methods.contains("encoder") || !methods.contains("text_decoder")) {
                throw new IllegalArgumentException(
                        "This app needs the optimized encoder/text_decoder export. Found "
                                + methods
                );
            }
            String nextBackend = String.join(
                    ", ",
                    nextModel.getMethodMetadata("encoder").getBackends()
            );
            nextModel.loadMethod("encoder");
            nextModel.loadMethod("text_decoder");

            model = nextModel;
            preprocessor = nextPreprocessor;
            tokenizer = nextTokenizer;
            backend = nextBackend;
            nextModel = null;
            nextPreprocessor = null;
        } finally {
            if (nextPreprocessor != null) {
                nextPreprocessor.close();
            }
            if (nextModel != null) {
                nextModel.close();
            }
        }
    }

    @Override
    public synchronized boolean isLoaded() {
        return model != null && preprocessor != null && tokenizer != null;
    }

    @Override
    public synchronized TranscriptionResult transcribe(
            float[] audio16Khz,
            ProgressListener progressListener
    ) {
        if (!isLoaded()) {
            throw new IllegalStateException("Load an ExecuTorch model before transcribing");
        }
        if (audio16Khz == null || audio16Khz.length == 0) {
            throw new IllegalArgumentException("Record some audio before transcribing");
        }
        ProgressListener progress = progressListener == null
                ? ProgressListener.NONE
                : progressListener;
        long started = System.nanoTime();

        progress.onProgress("Preparing the recording", 0.08f);
        Tensor waveformTensor = Tensor.fromBlob(
                audio16Khz,
                new long[]{audio16Khz.length}
        );
        String[] preprocessorMethods = preprocessor.getMethods();
        if (preprocessorMethods.length == 0) {
            throw new IllegalStateException("The audio preprocessor contains no callable method");
        }
        String preprocessorMethod = Arrays.asList(preprocessorMethods).contains("forward")
                ? "forward"
                : preprocessorMethods[0];
        Tensor features = preprocessor.execute(
                preprocessorMethod,
                EValue.from(waveformTensor)
        )[0].toTensor();

        progress.onProgress("Encoding speech", 0.25f);
        Tensor encoderHidden = model.execute(
                "encoder",
                EValue.from(features)
        )[0].toTensor();
        Log.i(TAG, "features=" + Arrays.toString(features.shape())
                + " encoderHidden=" + Arrays.toString(encoderHidden.shape()));

        progress.onProgress("Writing the transcript", 0.38f);
        List<Integer> tokens = new ArrayList<>();
        List<Integer> generatedTokens = new ArrayList<>();
        tokens.add(DECODER_START_TOKEN_ID);
        int cachePosition = 0;
        int forcedPrefixIndex = 0;

        for (int step = 0; step < MAX_GENERATION_TOKENS + FORCED_PREFIX_IDS.length; step++) {
            Tensor tokenTensor = Tensor.fromBlob(
                    new long[]{tokens.get(tokens.size() - 1)},
                    new long[]{1, 1}
            );
            Tensor positionTensor = Tensor.fromBlob(
                    new long[]{cachePosition},
                    new long[]{1}
            );
            Tensor logitsTensor = model.execute(
                    "text_decoder",
                    EValue.from(tokenTensor),
                    EValue.from(encoderHidden),
                    EValue.from(positionTensor)
            )[0].toTensor();
            float[] logits = logitsTensor.getDataAsFloatArray();
            if (step == 0) {
                Log.i(TAG, "decoderLogits=" + Arrays.toString(logitsTensor.shape()));
            }

            int nextToken;
            boolean generatedToken = false;
            if (forcedPrefixIndex < FORCED_PREFIX_IDS.length) {
                nextToken = FORCED_PREFIX_IDS[forcedPrefixIndex++];
            } else {
                nextToken = argmax(logits, tokens.size() == FORCED_PREFIX_IDS.length + 1);
                generatedTokens.add(nextToken);
                generatedToken = true;
            }

            tokens.add(nextToken);
            if (generatedToken && generatedTokens.size() <= 20) {
                Log.i(TAG, "token[" + generatedTokens.size() + "]=" + nextToken);
            }
            if (generatedToken && generatedTokens.size() % 8 == 0) {
                float fraction = 0.38f + Math.min(0.56f,
                        generatedTokens.size() / (float) MAX_GENERATION_TOKENS * 0.56f);
                progress.onProgress("Writing the transcript", fraction);
            }
            cachePosition++;
            if (nextToken == EOS_TOKEN_ID) {
                break;
            }
            int repeatedPatternLength = findRepeatedSuffixPattern(generatedTokens);
            if (generatedToken && repeatedPatternLength > 0) {
                int trimCount = repeatedPatternLength * REPETITION_GUARD_REPEATS;
                generatedTokens.subList(
                        generatedTokens.size() - trimCount,
                        generatedTokens.size()
                ).clear();
                tokens.subList(tokens.size() - trimCount, tokens.size()).clear();
                Log.w(TAG, "Stopped repeated token pattern of length " + repeatedPatternLength);
                break;
            }
        }

        long elapsedMillis = Math.round((System.nanoTime() - started) / 1_000_000.0);
        progress.onProgress("Transcript ready", 1.0f);
        return new TranscriptionResult(
                tokenizer.decode(tokens),
                elapsedMillis,
                tokens.size(),
                backend
        );
    }

    @Override
    public synchronized void close() {
        if (preprocessor != null) {
            preprocessor.close();
            preprocessor = null;
        }
        if (model != null) {
            model.close();
            model = null;
        }
        tokenizer = null;
        backend = "";
    }

    private static File requiredFile(File directory, String name) {
        if (directory == null || !directory.isDirectory()) {
            throw new IllegalArgumentException("The selected model directory is missing");
        }
        File file = new File(directory, name);
        if (!file.isFile()) {
            throw new IllegalArgumentException("The model package is missing " + name);
        }
        return file;
    }

    private static int findRepeatedSuffixPattern(List<Integer> tokenIds) {
        int maximumPatternLength = Math.min(
                REPETITION_GUARD_MAX_PATTERN,
                tokenIds.size() / REPETITION_GUARD_REPEATS
        );
        for (int patternLength = REPETITION_GUARD_MIN_PATTERN;
             patternLength <= maximumPatternLength;
             patternLength++) {
            int patternStart = tokenIds.size() - patternLength;
            List<Integer> pattern = tokenIds.subList(patternStart, tokenIds.size());
            boolean repeated = true;
            for (int repeat = 2; repeat <= REPETITION_GUARD_REPEATS; repeat++) {
                int start = tokenIds.size() - repeat * patternLength;
                int end = start + patternLength;
                if (!tokenIds.subList(start, end).equals(pattern)) {
                    repeated = false;
                    break;
                }
            }
            if (repeated) {
                return patternLength;
            }
        }
        return 0;
    }

    private static int argmax(float[] logits, boolean firstFreeToken) {
        int bestIndex = -1;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int index = 0; index < logits.length; index++) {
            boolean suppressed = SUPPRESSED.contains(index)
                    || (firstFreeToken && (index == 220 || index == EOS_TOKEN_ID));
            if (!suppressed && logits[index] > bestValue) {
                bestValue = logits[index];
                bestIndex = index;
            }
        }
        if (bestIndex < 0) {
            throw new IllegalStateException("The decoder did not produce a selectable token");
        }
        return bestIndex;
    }

    private static Set<Integer> createSuppressedSet() {
        Set<Integer> result = new HashSet<>();
        for (int token : SUPPRESS_TOKENS) {
            result.add(token);
        }
        return result;
    }
}
