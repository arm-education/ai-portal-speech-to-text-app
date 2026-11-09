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

/** Adapter for the three-program Whisper Base export with fixed decoder KV caches. */
public final class ExecuTorchStaticCacheWhisperAdapter implements SpeechToTextAdapter {
    private static final String TAG = "WhisperStaticCache";
    private static final String FORWARD = "forward";
    private static final String ENCODER_FILE = "encoder.pte";
    private static final String DECODER_PREFILL_FILE = "decoder_prefill.pte";
    private static final String DECODER_DECODE_FILE = "decoder_decode.pte";
    private static final int DECODER_LAYERS = 6;
    private static final int DECODER_CACHE_COUNT = DECODER_LAYERS * 2;
    private static final int PREFILL_SEMANTIC_OUTPUT_COUNT = 38;
    private static final int DECODE_MUTATION_OUTPUT_COUNT = DECODER_LAYERS * 3;
    private static final int VOCABULARY_SIZE = 51865;
    private static final int DECODER_START_TOKEN_ID = 50258;
    private static final int EOS_TOKEN_ID = 50257;
    private static final int[] PREFIX_TOKEN_IDS = {
            DECODER_START_TOKEN_ID, 50259, 50359, 50363
    };
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

    private Module encoder;
    private Module decoderPrefill;
    private Module decoderDecode;
    private Module preprocessor;
    private WhisperTokenizer tokenizer;
    private String backend = "";

    @Override
    public String id() {
        return AdapterRegistry.EXECUTORCH_STATIC_CACHE_ID;
    }

    @Override
    public synchronized void load(
            WhisperModelDescriptor descriptor,
            File modelDirectory
    ) throws Exception {
        if (descriptor == null || !id().equals(descriptor.adapterId())) {
            throw new IllegalArgumentException(
                    "This package is not a static-cache ExecuTorch model"
            );
        }

        File encoderFile = requiredFile(modelDirectory, ENCODER_FILE);
        File prefillFile = requiredFile(modelDirectory, DECODER_PREFILL_FILE);
        File decodeFile = requiredFile(modelDirectory, DECODER_DECODE_FILE);
        File preprocessorFile = requiredFile(modelDirectory, "whisper_preprocessor.pte");
        File tokenizerFile = requiredFile(modelDirectory, "tokenizer.json");

        close();
        Module nextEncoder = null;
        Module nextPrefill = null;
        Module nextDecode = null;
        Module nextPreprocessor = null;
        try {
            nextEncoder = loadForwardModule(encoderFile, 4);
            nextPrefill = loadForwardModule(prefillFile, 4);
            nextDecode = loadForwardModule(decodeFile, 4);
            nextPreprocessor = Module.load(
                    preprocessorFile.getAbsolutePath(),
                    Module.LOAD_MODE_MMAP,
                    2
            );
            requireMethod(nextPreprocessor, FORWARD, "audio preprocessor");
            nextPreprocessor.loadMethod(FORWARD);
            WhisperTokenizer nextTokenizer = new WhisperTokenizer(tokenizerFile);

            String nextBackend = String.join(
                    ", ",
                    nextEncoder.getMethodMetadata(FORWARD).getBackends()
            );

            encoder = nextEncoder;
            decoderPrefill = nextPrefill;
            decoderDecode = nextDecode;
            preprocessor = nextPreprocessor;
            tokenizer = nextTokenizer;
            backend = nextBackend;
            nextEncoder = null;
            nextPrefill = null;
            nextDecode = null;
            nextPreprocessor = null;
        } finally {
            closeModule(nextPreprocessor);
            closeModule(nextDecode);
            closeModule(nextPrefill);
            closeModule(nextEncoder);
        }
    }

    @Override
    public synchronized boolean isLoaded() {
        return encoder != null
                && decoderPrefill != null
                && decoderDecode != null
                && preprocessor != null
                && tokenizer != null;
    }

    @Override
    public synchronized TranscriptionResult transcribe(
            float[] audio16Khz,
            ProgressListener progressListener
    ) {
        if (!isLoaded()) {
            throw new IllegalStateException(
                    "Load the static-cache ExecuTorch model before transcribing"
            );
        }
        if (audio16Khz == null || audio16Khz.length == 0) {
            throw new IllegalArgumentException("Record some audio before transcribing");
        }
        ProgressListener progress = progressListener == null
                ? ProgressListener.NONE
                : progressListener;
        long started = System.nanoTime();

        progress.onProgress("Preparing the recording", 0.08f);
        Tensor waveform = Tensor.fromBlob(audio16Khz, new long[]{audio16Khz.length});
        Tensor features = preprocessor.execute(
                FORWARD,
                EValue.from(waveform)
        )[0].toTensor();

        progress.onProgress("Encoding speech", 0.25f);
        Tensor encoderHidden = encoder.execute(
                FORWARD,
                EValue.from(features)
        )[0].toTensor();
        Log.i(TAG, "features=" + Arrays.toString(features.shape())
                + " encoderHidden=" + Arrays.toString(encoderHidden.shape()));

        progress.onProgress("Writing the transcript", 0.38f);
        DecoderState state = prefill(encoderHidden);
        List<Integer> tokens = new ArrayList<>();
        for (int token : PREFIX_TOKEN_IDS) {
            tokens.add(token);
        }
        List<Integer> generatedTokens = new ArrayList<>();

        int nextToken = argmaxLastPosition(state.logits, true);
        for (int generatedCount = 0;
             generatedCount < MAX_GENERATION_TOKENS;
             generatedCount++) {
            tokens.add(nextToken);
            generatedTokens.add(nextToken);
            if (generatedTokens.size() <= 20) {
                Log.i(TAG, "token[" + generatedTokens.size() + "]=" + nextToken);
            }
            if (nextToken == EOS_TOKEN_ID) {
                break;
            }

            int repeatedPatternLength = findRepeatedSuffixPattern(generatedTokens);
            if (repeatedPatternLength > 0) {
                int trimCount = repeatedPatternLength * REPETITION_GUARD_REPEATS;
                generatedTokens.subList(
                        generatedTokens.size() - trimCount,
                        generatedTokens.size()
                ).clear();
                tokens.subList(tokens.size() - trimCount, tokens.size()).clear();
                Log.w(TAG, "Stopped repeated token pattern of length "
                        + repeatedPatternLength);
                break;
            }

            if (generatedTokens.size() % 8 == 0) {
                float fraction = 0.38f + Math.min(
                        0.56f,
                        generatedTokens.size() / (float) MAX_GENERATION_TOKENS * 0.56f
                );
                progress.onProgress("Writing the transcript", fraction);
            }
            nextToken = decode(nextToken, state);
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
        closeModule(preprocessor);
        preprocessor = null;
        closeModule(decoderDecode);
        decoderDecode = null;
        closeModule(decoderPrefill);
        decoderPrefill = null;
        closeModule(encoder);
        encoder = null;
        tokenizer = null;
        backend = "";
    }

    private DecoderState prefill(Tensor encoderHidden) {
        List<EValue> inputs = new ArrayList<>();
        inputs.add(EValue.from(Tensor.fromBlob(
                new long[]{
                        PREFIX_TOKEN_IDS[0],
                        PREFIX_TOKEN_IDS[1],
                        PREFIX_TOKEN_IDS[2],
                        PREFIX_TOKEN_IDS[3]
                },
                new long[]{1, PREFIX_TOKEN_IDS.length}
        )));
        inputs.add(EValue.from(emptyCacheLength()));
        inputs.add(EValue.from(encoderHidden));
        for (int cache = 1; cache < DECODER_CACHE_COUNT; cache++) {
            inputs.add(EValue.from(emptyCacheLength()));
        }

        EValue[] outputs = decoderPrefill.execute(
                FORWARD,
                inputs.toArray(new EValue[0])
        );
        if (outputs.length < PREFILL_SEMANTIC_OUTPUT_COUNT) {
            throw new IllegalStateException(
                    "Decoder prefill returned " + outputs.length
                            + " outputs; expected at least "
                            + PREFILL_SEMANTIC_OUTPUT_COUNT
            );
        }
        int semanticOffset = outputs.length - PREFILL_SEMANTIC_OUTPUT_COUNT;
        DecoderState state = new DecoderState();
        for (int layer = 0; layer < DECODER_LAYERS; layer++) {
            int selfOffset = semanticOffset + 1 + layer * 3;
            state.selfLengths[layer] = outputs[selfOffset].toTensor();
            state.selfKeys[layer] = outputs[selfOffset + 1].toTensor();
            state.selfValues[layer] = outputs[selfOffset + 2].toTensor();

            int crossOffset = semanticOffset + 19 + layer * 3;
            state.crossKeys[layer] = outputs[crossOffset + 1].toTensor();
            state.crossValues[layer] = outputs[crossOffset + 2].toTensor();
        }
        state.logits = outputs[outputs.length - 1].toTensor();
        return state;
    }

    private int decode(int token, DecoderState state) {
        List<EValue> inputs = new ArrayList<>();
        inputs.add(EValue.from(Tensor.fromBlob(
                new long[]{token},
                new long[]{1, 1}
        )));
        for (int layer = 0; layer < DECODER_LAYERS; layer++) {
            inputs.add(EValue.from(state.selfLengths[layer]));
            inputs.add(EValue.from(state.selfKeys[layer]));
            inputs.add(EValue.from(state.selfValues[layer]));
            inputs.add(EValue.from(state.crossKeys[layer]));
            inputs.add(EValue.from(state.crossValues[layer]));
        }

        EValue[] outputs = decoderDecode.execute(
                FORWARD,
                inputs.toArray(new EValue[0])
        );
        if (outputs.length == DECODE_MUTATION_OUTPUT_COUNT + 1) {
            for (int layer = 0; layer < DECODER_LAYERS; layer++) {
                int mutationOffset = layer * 3;
                state.selfLengths[layer] = outputs[mutationOffset].toTensor();
                state.selfKeys[layer] = outputs[mutationOffset + 1].toTensor();
                state.selfValues[layer] = outputs[mutationOffset + 2].toTensor();
            }
        } else if (outputs.length != 1) {
            throw new IllegalStateException(
                    "Decoder returned " + outputs.length + " outputs; expected 1 or "
                            + (DECODE_MUTATION_OUTPUT_COUNT + 1)
            );
        }
        state.logits = outputs[outputs.length - 1].toTensor();
        return argmaxLastPosition(state.logits, false);
    }

    private static Tensor emptyCacheLength() {
        return Tensor.fromBlob(new long[]{0L}, new long[0]);
    }

    private static int argmaxLastPosition(Tensor logitsTensor, boolean firstFreeToken) {
        float[] logits = logitsTensor.getDataAsFloatArray();
        if (logits.length < VOCABULARY_SIZE
                || logits.length % VOCABULARY_SIZE != 0) {
            throw new IllegalStateException(
                    "Unexpected decoder logits shape "
                            + Arrays.toString(logitsTensor.shape())
            );
        }
        int offset = logits.length - VOCABULARY_SIZE;
        int bestIndex = -1;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int index = 0; index < VOCABULARY_SIZE; index++) {
            boolean suppressed = SUPPRESSED.contains(index)
                    || (firstFreeToken && (index == 220 || index == EOS_TOKEN_ID));
            if (!suppressed && logits[offset + index] > bestValue) {
                bestValue = logits[offset + index];
                bestIndex = index;
            }
        }
        if (bestIndex < 0) {
            throw new IllegalStateException("The decoder did not produce a selectable token");
        }
        return bestIndex;
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

    private static Module loadForwardModule(File file, int threads) {
        Module module = Module.load(file.getAbsolutePath(), Module.LOAD_MODE_MMAP, threads);
        try {
            requireMethod(module, FORWARD, file.getName());
            module.loadMethod(FORWARD);
            return module;
        } catch (RuntimeException error) {
            module.close();
            throw error;
        }
    }

    private static void requireMethod(Module module, String method, String description) {
        if (!Arrays.asList(module.getMethods()).contains(method)) {
            throw new IllegalArgumentException(
                    description + " does not contain a " + method + " method"
            );
        }
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

    private static void closeModule(Module module) {
        if (module != null) {
            module.close();
        }
    }

    private static Set<Integer> createSuppressedSet() {
        Set<Integer> result = new HashSet<>();
        for (int token : SUPPRESS_TOKENS) {
            result.add(token);
        }
        return result;
    }

    private static final class DecoderState {
        final Tensor[] selfLengths = new Tensor[DECODER_LAYERS];
        final Tensor[] selfKeys = new Tensor[DECODER_LAYERS];
        final Tensor[] selfValues = new Tensor[DECODER_LAYERS];
        final Tensor[] crossKeys = new Tensor[DECODER_LAYERS];
        final Tensor[] crossValues = new Tensor[DECODER_LAYERS];
        Tensor logits;
    }
}
