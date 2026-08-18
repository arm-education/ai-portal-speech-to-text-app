package org.arm.learningpath.whisper.litert;

import org.arm.learningpath.whisper.AdapterRegistry;
import org.arm.learningpath.whisper.ProgressListener;
import org.arm.learningpath.whisper.SpeechToTextAdapter;
import org.arm.learningpath.whisper.TranscriptionResult;
import org.arm.learningpath.whisper.WhisperModelDescriptor;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Runs the Arm Whisper Base, Medium, and Large V3 multi-signature LiteRT exports. */
public final class LiteRtWhisperAdapter implements SpeechToTextAdapter {
    private static final float MASKED_LOGIT = -1.0e30f;
    private static final int MAX_PACKAGE_DEPTH = 8;

    private Interpreter interpreter;
    private WhisperModelDescriptor descriptor;
    private DecoderProfile profile;
    private ModelLayout layout;
    private WhisperTokenizer tokenizer;

    @Override
    public String id() {
        return AdapterRegistry.LITERT_ID;
    }

    @Override
    public synchronized void load(
            WhisperModelDescriptor selectedDescriptor,
            File modelDirectory
    ) throws Exception {
        close();
        if (selectedDescriptor == null) {
            throw new IllegalArgumentException("Select a LiteRT Whisper model first");
        }
        if (!id().equals(selectedDescriptor.adapterId())) {
            throw new IllegalArgumentException(
                    selectedDescriptor.displayName() + " does not use the LiteRT adapter"
            );
        }
        if (modelDirectory == null || !modelDirectory.isDirectory()) {
            throw new IllegalArgumentException("Select an extracted model package directory");
        }

        DecoderProfile selectedProfile = DecoderProfile.forModel(selectedDescriptor);
        File modelFile = findExactlyOne(modelDirectory, ".tflite");
        File tokenizerFile = findExactlyNamed(modelDirectory, "tokenizer.json");

        Interpreter.Options options = new Interpreter.Options();
        options.setUseXNNPACK(true);
        options.setNumThreads(Math.max(
                1,
                Math.min(4, Runtime.getRuntime().availableProcessors())
        ));

        Interpreter candidate = null;
        try {
            candidate = new Interpreter(modelFile, options);
            candidate.allocateTensors();
            ModelLayout inspected = ModelLayout.inspect(candidate, selectedProfile);
            WhisperTokenizer parsedTokenizer = new WhisperTokenizer(tokenizerFile);

            interpreter = candidate;
            descriptor = selectedDescriptor;
            profile = selectedProfile;
            layout = inspected;
            tokenizer = parsedTokenizer;
            candidate = null;
        } finally {
            if (candidate != null) {
                candidate.close();
            }
        }
    }

    @Override
    public synchronized boolean isLoaded() {
        return interpreter != null;
    }

    @Override
    public synchronized TranscriptionResult transcribe(
            float[] audio16Khz,
            ProgressListener progressListener
    ) throws Exception {
        requireLoaded();
        ProgressListener progress = progressListener == null
                ? ProgressListener.NONE
                : progressListener;
        long started = System.nanoTime();

        progress.onProgress("Preparing 16 kHz audio", 0.03f);
        ByteBuffer features = WhisperFeatureExtractor.extract(
                audio16Khz,
                layout.melBins,
                layout.featureFrames
        );

        progress.onProgress("Running " + descriptor.displayName() + " encoder", 0.20f);
        List<ByteBuffer> crossCaches = encode(features);

        progress.onProgress("Decoding speech", 0.45f);
        List<Integer> generatedTokens = decode(crossCaches, progress);

        progress.onProgress("Finishing transcript", 0.98f);
        String text = tokenizer.decode(generatedTokens);
        long elapsedMillis = Math.round((System.nanoTime() - started) / 1_000_000.0);
        progress.onProgress("Transcript ready", 1.0f);
        return new TranscriptionResult(
                text,
                elapsedMillis,
                generatedTokens.size(),
                "LiteRT / XNNPACK"
        );
    }

    private List<ByteBuffer> encode(ByteBuffer features) {
        Map<String, Object> inputs = Collections.singletonMap(layout.encoderInputName, features);
        LinkedHashMap<String, Object> outputs = new LinkedHashMap<>();
        List<ByteBuffer> buffers = new ArrayList<>(layout.encoderOutputNames.size());
        for (String name : layout.encoderOutputNames) {
            Tensor tensor = interpreter.getOutputTensorFromSignature(name, "encode");
            ByteBuffer buffer = directBuffer(tensor.numBytes());
            buffers.add(buffer);
            outputs.put(name, buffer);
        }

        interpreter.runSignature(inputs, outputs, "encode");
        for (ByteBuffer buffer : buffers) {
            buffer.rewind();
        }
        return buffers;
    }

    private List<Integer> decode(
            List<ByteBuffer> crossCaches,
            ProgressListener progress
    ) {
        List<ByteBuffer> selfInputs = allocateDecodeBuffers(layout.selfInputNames, true);
        List<ByteBuffer> selfOutputs = allocateDecodeBuffers(layout.selfOutputNames, false);
        ByteBuffer logits = directBuffer(
                interpreter.getOutputTensorFromSignature(layout.logitsName, "decode").numBytes()
        );
        ByteBuffer tokenBuffer = directBuffer(Long.BYTES);
        ByteBuffer positionBuffer = directBuffer(Long.BYTES);
        ByteBuffer maskBuffer = directBuffer(layout.maxPositions * Float.BYTES);
        boolean[] suppressed = blockedTokens(profile.suppressTokens, layout.vocabularySize);
        boolean[] beginSuppressed = blockedTokens(
                profile.beginSuppressTokens,
                layout.vocabularySize
        );

        int token = profile.promptIds[0];
        int maxSteps = Math.min(
                layout.maxPositions,
                profile.promptIds.length + profile.maxNewTokens
        );
        List<Integer> generated = new ArrayList<>(profile.maxNewTokens);

        for (int position = 0; position < maxSteps; position++) {
            tokenBuffer.putLong(0, token);
            positionBuffer.putLong(0, position);
            writeAttentionMask(maskBuffer, position, layout.maxPositions);

            LinkedHashMap<String, Object> inputs = new LinkedHashMap<>();
            inputs.put(layout.tokenName, tokenBuffer);
            inputs.put(layout.positionName, positionBuffer);
            inputs.put(layout.maskName, maskBuffer);
            for (int index = 0; index < selfInputs.size(); index++) {
                ByteBuffer buffer = selfInputs.get(index);
                buffer.rewind();
                inputs.put(layout.selfInputNames.get(index), buffer);
            }
            for (int index = 0; index < crossCaches.size(); index++) {
                ByteBuffer buffer = crossCaches.get(index);
                buffer.rewind();
                inputs.put(layout.crossInputNames.get(index), buffer);
            }

            logits.rewind();
            LinkedHashMap<String, Object> outputs = new LinkedHashMap<>();
            outputs.put(layout.logitsName, logits);
            for (int index = 0; index < selfOutputs.size(); index++) {
                ByteBuffer buffer = selfOutputs.get(index);
                buffer.rewind();
                outputs.put(layout.selfOutputNames.get(index), buffer);
            }
            interpreter.runSignature(inputs, outputs, "decode");

            List<ByteBuffer> previousInputs = selfInputs;
            selfInputs = selfOutputs;
            selfOutputs = previousInputs;

            if (position < profile.promptIds.length - 1) {
                token = profile.promptIds[position + 1];
                continue;
            }

            int next = argmax(
                    logits,
                    layout.vocabularySize,
                    suppressed,
                    generated.isEmpty() ? beginSuppressed : null
            );
            if (next == profile.eosTokenId) {
                break;
            }
            generated.add(next);
            if (generated.size() >= profile.maxNewTokens) {
                break;
            }
            token = next;

            if (generated.size() == 1 || generated.size() % 8 == 0) {
                float fraction = 0.45f
                        + 0.50f * generated.size() / profile.maxNewTokens;
                progress.onProgress("Decoding speech", fraction);
            }
        }
        return generated;
    }

    private List<ByteBuffer> allocateDecodeBuffers(List<String> names, boolean input) {
        List<ByteBuffer> buffers = new ArrayList<>(names.size());
        for (String name : names) {
            Tensor tensor = input
                    ? interpreter.getInputTensorFromSignature(name, "decode")
                    : interpreter.getOutputTensorFromSignature(name, "decode");
            buffers.add(directBuffer(tensor.numBytes()));
        }
        return buffers;
    }

    private static int argmax(
            ByteBuffer logits,
            int vocabularySize,
            boolean[] suppressed,
            boolean[] beginSuppressed
    ) {
        int bestToken = -1;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (int token = 0; token < vocabularySize; token++) {
            if (suppressed[token]
                    || (beginSuppressed != null && beginSuppressed[token])) {
                continue;
            }
            float score = logits.getFloat(token * Float.BYTES);
            if (!Float.isNaN(score) && (bestToken < 0 || score > bestScore)) {
                bestToken = token;
                bestScore = score;
            }
        }
        if (bestToken < 0) {
            throw new IllegalStateException("The decoder returned no usable token logits");
        }
        return bestToken;
    }

    private static boolean[] blockedTokens(int[] tokenIds, int vocabularySize) {
        boolean[] result = new boolean[vocabularySize];
        for (int tokenId : tokenIds) {
            if (tokenId >= 0 && tokenId < vocabularySize) {
                result[tokenId] = true;
            }
        }
        return result;
    }

    private static void writeAttentionMask(
            ByteBuffer buffer,
            int position,
            int maxPositions
    ) {
        buffer.clear();
        for (int index = 0; index < maxPositions; index++) {
            buffer.putFloat(index <= position ? 0.0f : MASKED_LOGIT);
        }
        buffer.rewind();
    }

    private void requireLoaded() {
        if (interpreter == null || descriptor == null || profile == null
                || layout == null || tokenizer == null) {
            throw new IllegalStateException("Import a LiteRT Whisper package first");
        }
    }

    @Override
    public synchronized void close() {
        if (interpreter != null) {
            interpreter.close();
        }
        interpreter = null;
        descriptor = null;
        profile = null;
        layout = null;
        tokenizer = null;
    }

    private static File findExactlyOne(File root, String suffix) throws IOException {
        List<File> matches = new ArrayList<>();
        collectMatches(root, suffix.toLowerCase(Locale.ROOT), null, 0, matches);
        if (matches.size() != 1) {
            throw new IOException(
                    "Expected exactly one " + suffix + " file, found " + matches.size()
            );
        }
        return matches.get(0);
    }

    private static File findExactlyNamed(File root, String fileName) throws IOException {
        List<File> matches = new ArrayList<>();
        collectMatches(root, null, fileName.toLowerCase(Locale.ROOT), 0, matches);
        if (matches.size() != 1) {
            throw new IOException(
                    "Expected exactly one " + fileName + ", found " + matches.size()
            );
        }
        return matches.get(0);
    }

    private static void collectMatches(
            File current,
            String suffix,
            String exactName,
            int depth,
            List<File> matches
    ) throws IOException {
        if (depth > MAX_PACKAGE_DEPTH) {
            throw new IOException("The model package is nested too deeply");
        }
        File[] children = current.listFiles();
        if (children == null) {
            throw new IOException("Could not read " + current);
        }
        for (File child : children) {
            if (child.isDirectory()) {
                if (!".cache".equals(child.getName())) {
                    collectMatches(child, suffix, exactName, depth + 1, matches);
                }
                continue;
            }
            String lowerName = child.getName().toLowerCase(Locale.ROOT);
            if ((suffix != null && lowerName.endsWith(suffix))
                    || (exactName != null && lowerName.equals(exactName))) {
                matches.add(child);
            }
        }
    }

    private static ByteBuffer directBuffer(int byteCount) {
        return ByteBuffer.allocateDirect(byteCount).order(ByteOrder.nativeOrder());
    }

    private static int compareTensorNames(String left, String right) {
        int suffixCompare = Integer.compare(numericSuffix(left), numericSuffix(right));
        if (suffixCompare != 0) {
            return suffixCompare;
        }
        int roleCompare = Integer.compare(cacheRole(left), cacheRole(right));
        return roleCompare != 0 ? roleCompare : left.compareTo(right);
    }

    private static int cacheRole(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        if (lower.startsWith("k") || lower.contains("_k")) {
            return 0;
        }
        if (lower.startsWith("v") || lower.contains("_v")) {
            return 1;
        }
        return 2;
    }

    private static int numericSuffix(String name) {
        int end = name.length();
        int start = end;
        while (start > 0 && Character.isDigit(name.charAt(start - 1))) {
            start--;
        }
        if (start == end) {
            return Integer.MAX_VALUE;
        }
        try {
            return Integer.parseInt(name.substring(start));
        } catch (NumberFormatException ignored) {
            return Integer.MAX_VALUE;
        }
    }

    private static final class ModelLayout {
        final String encoderInputName;
        final List<String> encoderOutputNames;
        final String tokenName;
        final String positionName;
        final String maskName;
        final List<String> selfInputNames;
        final List<String> crossInputNames;
        final String logitsName;
        final List<String> selfOutputNames;
        final int melBins;
        final int featureFrames;
        final int maxPositions;
        final int vocabularySize;

        ModelLayout(
                String encoderInputName,
                List<String> encoderOutputNames,
                String tokenName,
                String positionName,
                String maskName,
                List<String> selfInputNames,
                List<String> crossInputNames,
                String logitsName,
                List<String> selfOutputNames,
                int melBins,
                int featureFrames,
                int maxPositions,
                int vocabularySize
        ) {
            this.encoderInputName = encoderInputName;
            this.encoderOutputNames = encoderOutputNames;
            this.tokenName = tokenName;
            this.positionName = positionName;
            this.maskName = maskName;
            this.selfInputNames = selfInputNames;
            this.crossInputNames = crossInputNames;
            this.logitsName = logitsName;
            this.selfOutputNames = selfOutputNames;
            this.melBins = melBins;
            this.featureFrames = featureFrames;
            this.maxPositions = maxPositions;
            this.vocabularySize = vocabularySize;
        }

        static ModelLayout inspect(Interpreter active, DecoderProfile profile) {
            Set<String> signatures = new HashSet<>(Arrays.asList(active.getSignatureKeys()));
            if (!signatures.contains("encode") || !signatures.contains("decode")) {
                throw unsupported("model must expose encode and decode signatures");
            }

            String[] encoderInputs = active.getSignatureInputs("encode");
            if (encoderInputs.length != 1) {
                throw unsupported("encode must have exactly one input");
            }
            Tensor featureTensor = active.getInputTensorFromSignature(
                    encoderInputs[0],
                    "encode"
            );
            int[] featureShape = featureTensor.shape();
            if (featureTensor.dataType() != DataType.FLOAT32
                    || featureShape.length != 3
                    || featureShape[0] != 1
                    || featureShape[1] != profile.melBins
                    || featureShape[2] <= 0) {
                throw unsupported(
                        "encode input must be FLOAT32 [1, " + profile.melBins
                                + ", frames], found " + describe(featureTensor)
                );
            }

            List<String> encoderOutputs = sorted(active.getSignatureOutputs("encode"));
            if (encoderOutputs.size() != profile.cacheTensorCount()) {
                throw unsupported(
                        "encode must produce " + profile.cacheTensorCount()
                                + " cross-attention cache tensors"
                );
            }

            String token = null;
            String position = null;
            String mask = null;
            List<String> selfInputs = new ArrayList<>();
            List<String> crossInputs = new ArrayList<>();
            for (String name : sorted(active.getSignatureInputs("decode"))) {
                Tensor tensor = active.getInputTensorFromSignature(name, "decode");
                int[] shape = tensor.shape();
                if (tensor.dataType() == DataType.INT64
                        && Arrays.equals(shape, new int[]{1, 1})) {
                    if (token != null) {
                        throw unsupported("decode contains more than one token input");
                    }
                    token = name;
                } else if (tensor.dataType() == DataType.INT64
                        && Arrays.equals(shape, new int[]{1})) {
                    if (position != null) {
                        throw unsupported("decode contains more than one position input");
                    }
                    position = name;
                } else if (tensor.dataType() == DataType.FLOAT32
                        && Arrays.equals(
                                shape,
                                new int[]{1, 1, 1, profile.maxTargetPositions}
                        )) {
                    if (mask != null) {
                        throw unsupported("decode contains more than one attention mask");
                    }
                    mask = name;
                } else if (isCacheTensor(tensor, profile, profile.maxTargetPositions)) {
                    selfInputs.add(name);
                } else if (isCrossCacheTensor(tensor, profile)) {
                    crossInputs.add(name);
                } else {
                    throw unsupported(
                            "unrecognized decode input " + name + " " + describe(tensor)
                    );
                }
            }

            String logits = null;
            List<String> selfOutputs = new ArrayList<>();
            for (String name : sorted(active.getSignatureOutputs("decode"))) {
                Tensor tensor = active.getOutputTensorFromSignature(name, "decode");
                if (tensor.dataType() == DataType.FLOAT32
                        && Arrays.equals(
                                tensor.shape(),
                                new int[]{1, 1, profile.vocabularySize}
                        )) {
                    if (logits != null) {
                        throw unsupported("decode contains more than one logits output");
                    }
                    logits = name;
                } else if (isCacheTensor(tensor, profile, profile.maxTargetPositions)) {
                    selfOutputs.add(name);
                } else {
                    throw unsupported(
                            "unrecognized decode output " + name + " " + describe(tensor)
                    );
                }
            }

            if (token == null || position == null || mask == null || logits == null) {
                throw unsupported("decode is missing token, position, mask, or logits");
            }
            int expectedCaches = profile.cacheTensorCount();
            if (selfInputs.size() != expectedCaches
                    || crossInputs.size() != expectedCaches
                    || selfOutputs.size() != expectedCaches) {
                throw unsupported(
                        "expected " + expectedCaches + " self/cross cache tensors, found "
                                + selfInputs.size() + "/" + crossInputs.size() + "/"
                                + selfOutputs.size()
                );
            }

            validateCachePairs(active, encoderOutputs, crossInputs, "encoder/cross cache");
            validateCachePairs(active, selfOutputs, selfInputs, "self cache");
            return new ModelLayout(
                    encoderInputs[0],
                    encoderOutputs,
                    token,
                    position,
                    mask,
                    selfInputs,
                    crossInputs,
                    logits,
                    selfOutputs,
                    featureShape[1],
                    featureShape[2],
                    profile.maxTargetPositions,
                    profile.vocabularySize
            );
        }

        private static void validateCachePairs(
                Interpreter active,
                List<String> outputNames,
                List<String> inputNames,
                String label
        ) {
            for (int index = 0; index < outputNames.size(); index++) {
                String outputSignature = label.startsWith("encoder") ? "encode" : "decode";
                Tensor output = active.getOutputTensorFromSignature(
                        outputNames.get(index),
                        outputSignature
                );
                Tensor input = active.getInputTensorFromSignature(
                        inputNames.get(index),
                        "decode"
                );
                if (output.dataType() != input.dataType()
                        || !Arrays.equals(output.shape(), input.shape())
                        || output.numBytes() != input.numBytes()) {
                    throw unsupported(label + " tensor " + index + " does not match");
                }
            }
        }

        private static boolean isCacheTensor(
                Tensor tensor,
                DecoderProfile profile,
                int sequenceLength
        ) {
            return tensor.dataType() == DataType.FLOAT32
                    && Arrays.equals(
                            tensor.shape(),
                            new int[]{1, sequenceLength, profile.heads, profile.headDimension()}
                    );
        }

        private static boolean isCrossCacheTensor(Tensor tensor, DecoderProfile profile) {
            int[] shape = tensor.shape();
            return tensor.dataType() == DataType.FLOAT32
                    && shape.length == 4
                    && shape[0] == 1
                    && shape[1] > profile.maxTargetPositions
                    && shape[2] == profile.heads
                    && shape[3] == profile.headDimension();
        }

        private static String describe(Tensor tensor) {
            return tensor.dataType() + " " + Arrays.toString(tensor.shape())
                    + " (shape signature " + Arrays.toString(tensor.shapeSignature()) + ")";
        }

        private static List<String> sorted(String[] names) {
            List<String> result = new ArrayList<>(Arrays.asList(names));
            result.sort(LiteRtWhisperAdapter::compareTensorNames);
            return result;
        }

        private static IllegalArgumentException unsupported(String detail) {
            return new IllegalArgumentException("Unsupported LiteRT Whisper package: " + detail);
        }
    }

    private static final class DecoderProfile {
        private static final int[] COMMON_SUPPRESS = {
                1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61,
                62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918,
                922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536,
                3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428,
                10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265,
                15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
                26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
                47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362
        };
        private static final int[] LARGE_V3_SUPPRESS = {
                1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61,
                62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918,
                922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536,
                3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428,
                10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265,
                15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
                26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
                47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363
        };
        private static final int[] BEGIN_SUPPRESS = {220, 50257};

        final int layers;
        final int heads;
        final int modelWidth;
        final int melBins;
        final int vocabularySize;
        final int maxTargetPositions;
        final int maxNewTokens;
        final int eosTokenId;
        final int[] promptIds;
        final int[] suppressTokens;
        final int[] beginSuppressTokens;

        DecoderProfile(
                int layers,
                int heads,
                int modelWidth,
                int melBins,
                int vocabularySize,
                int[] promptIds,
                int[] suppressTokens
        ) {
            this.layers = layers;
            this.heads = heads;
            this.modelWidth = modelWidth;
            this.melBins = melBins;
            this.vocabularySize = vocabularySize;
            this.maxTargetPositions = 448;
            this.maxNewTokens = 128;
            this.eosTokenId = 50257;
            this.promptIds = promptIds;
            this.suppressTokens = suppressTokens;
            this.beginSuppressTokens = BEGIN_SUPPRESS;
        }

        int cacheTensorCount() {
            return layers * 2;
        }

        int headDimension() {
            return modelWidth / heads;
        }

        static DecoderProfile forModel(WhisperModelDescriptor descriptor) {
            DecoderProfile result;
            switch (descriptor.id()) {
                case "whisper-base-litert":
                    result = new DecoderProfile(
                            6, 8, 512, 80, 51865,
                            new int[]{50258, 50259, 50359, 50363},
                            COMMON_SUPPRESS
                    );
                    break;
                case "whisper-medium-litert":
                    result = new DecoderProfile(
                            24, 16, 1024, 80, 51865,
                            new int[]{50258, 50259, 50359, 50363},
                            COMMON_SUPPRESS
                    );
                    break;
                case "whisper-large-v3-litert":
                    result = new DecoderProfile(
                            32, 20, 1280, 128, 51866,
                            new int[]{50258, 50259, 50360, 50364},
                            LARGE_V3_SUPPRESS
                    );
                    break;
                default:
                    throw new IllegalArgumentException(
                            "The LiteRT adapter has no decoder profile for " + descriptor.id()
                    );
            }
            if (descriptor.melBins() != result.melBins
                    || descriptor.vocabularySize() != result.vocabularySize) {
                throw new IllegalArgumentException(
                        "The model descriptor dimensions do not match its LiteRT profile"
                );
            }
            return result;
        }
    }
}
