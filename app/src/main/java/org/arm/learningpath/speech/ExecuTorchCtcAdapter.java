package org.arm.learningpath.speech;

import org.json.JSONArray;
import org.json.JSONObject;
import org.pytorch.executorch.DType;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Iterator;

/** ExecuTorch runner for configured waveform-to-logits CTC exports. */
public final class ExecuTorchCtcAdapter implements SpeechToTextAdapter {
    public static final String CONFIG_CONFORMER_LARGE = "wav2vec2-conformer-large-960h-ctc";
    private static final String FORWARD = "forward";
    private static final int NUM_THREADS = 4;
    private static final int SAMPLE_RATE = 16000;

    private Module model;
    private Profile profile;
    private CtcDecoder decoder;
    private String backend = "";

    @Override
    public String id() {
        return AdapterRegistry.EXECUTORCH_CTC_ID;
    }

    @Override
    public synchronized void load(SpeechModelDescriptor descriptor, File modelDirectory)
            throws Exception {
        if (descriptor == null || !id().equals(descriptor.adapterId())) {
            throw new IllegalArgumentException("This package is not an ExecuTorch CTC model");
        }
        Profile nextProfile = Profile.forId(descriptor.configurationId());
        if (descriptor.melBins() != 0
                || descriptor.vocabularySize() != nextProfile.vocabularySize) {
            throw new IllegalArgumentException("The CTC model descriptor does not match its profile");
        }
        File modelFile = requiredFile(modelDirectory, "model.pte");
        CtcDecoder nextDecoder = readDecoder(modelDirectory, nextProfile);

        close();
        Module nextModel = null;
        try {
            nextModel = Module.load(modelFile.getAbsolutePath(), Module.LOAD_MODE_MMAP, NUM_THREADS);
            if (!Arrays.asList(nextModel.getMethods()).contains(FORWARD)) {
                throw new IllegalArgumentException("The CTC model has no forward method");
            }
            nextModel.loadMethod(FORWARD);
            String nextBackend = String.join(", ", nextModel.getMethodMetadata(FORWARD).getBackends());
            model = nextModel;
            profile = nextProfile;
            decoder = nextDecoder;
            backend = nextBackend;
            nextModel = null;
        } finally {
            if (nextModel != null) {
                nextModel.close();
            }
        }
    }

    @Override
    public synchronized boolean isLoaded() {
        return model != null && profile != null && decoder != null;
    }

    @Override
    public synchronized TranscriptionResult transcribe(float[] audio16Khz, ProgressListener listener) {
        if (!isLoaded()) {
            throw new IllegalStateException("Load the CTC model before transcribing");
        }
        if (audio16Khz == null || audio16Khz.length == 0
                || audio16Khz.length > profile.inputSamples) {
            throw new IllegalArgumentException("Record between half a second and 30 seconds of audio");
        }
        int validFrames = CtcAudioProcessor.outputFrames(
                audio16Khz.length, profile.convKernels, profile.convStrides);
        if (validFrames == 0) {
            throw new IllegalArgumentException("The recording is too short for the CTC model");
        }
        ProgressListener progress = listener == null ? ProgressListener.NONE : listener;
        long started = System.nanoTime();
        progress.onProgress("Preparing the recording", 0.1f);
        float[] input = CtcAudioProcessor.normalizeAndPad(audio16Khz, profile.inputSamples);
        Tensor waveform = Tensor.fromBlob(input, new long[]{1, profile.inputSamples});

        progress.onProgress("Recognizing speech", 0.25f);
        EValue[] outputs = model.execute(FORWARD, EValue.from(waveform));
        if (outputs.length != 1 || !outputs[0].isTensor()) {
            throw new IllegalStateException("The CTC model must return one tensor of frame scores");
        }
        Tensor scores = outputs[0].toTensor();
        long[] expectedShape = {1, profile.outputFrames, profile.vocabularySize};
        if (scores.dtype() != DType.FLOAT || !Arrays.equals(scores.shape(), expectedShape)) {
            throw new IllegalStateException("Unexpected CTC output: " + scores.dtype() + " "
                    + Arrays.toString(scores.shape()));
        }
        progress.onProgress("Writing the transcript", 0.9f);
        String text = decoder.decode(scores.getDataAsFloatArray(), validFrames);
        long elapsedMillis = Math.round((System.nanoTime() - started) / 1_000_000.0);
        progress.onProgress("Transcript ready", 1.0f);
        return new TranscriptionResult(text, elapsedMillis, backend);
    }

    @Override
    public synchronized void close() {
        if (model != null) {
            model.close();
        }
        model = null;
        profile = null;
        decoder = null;
        backend = "";
    }

    private static CtcDecoder readDecoder(File directory, Profile profile) throws Exception {
        JSONObject config = readJson(directory, "config.json");
        JSONObject processor = readJson(directory, "preprocessor_config.json");
        JSONObject tokenizer = readJson(directory, "tokenizer_config.json");
        JSONObject vocabulary = readJson(directory, "vocab.json");
        JSONArray architectures = config.getJSONArray("architectures");
        require("wav2vec2-conformer".equals(config.getString("model_type"))
                        && architectures.length() == 1
                        && "Wav2Vec2ConformerForCTC".equals(architectures.getString(0)),
                "Expected the Wav2Vec2 Conformer CTC configuration");
        require(!config.optBoolean("add_adapter", false), "CTC output adapters are not supported");
        require(config.getInt("vocab_size") == profile.vocabularySize,
                "The CTC vocabulary size does not match the model profile");
        require(matches(config.getJSONArray("conv_kernel"), profile.convKernels)
                        && matches(config.getJSONArray("conv_stride"), profile.convStrides),
                "The CTC convolution lengths do not match the model profile");
        require("Wav2Vec2FeatureExtractor".equals(processor.getString("feature_extractor_type"))
                        && processor.getInt("sampling_rate") == SAMPLE_RATE
                        && processor.getInt("feature_size") == 1
                        && processor.getBoolean("do_normalize")
                        && "right".equals(processor.getString("padding_side"))
                        && processor.getDouble("padding_value") == 0,
                "The CTC audio preprocessing configuration is incompatible");
        require("Wav2Vec2CTCTokenizer".equals(tokenizer.getString("tokenizer_class")),
                "Expected a Wav2Vec2 CTC tokenizer");
        require(" ".equals(tokenizer.optString("replace_word_delimiter_char", " ")),
                "The CTC tokenizer must use spaces between words");
        require(vocabulary.length() == profile.vocabularySize,
                "The CTC vocabulary does not contain the expected number of symbols");

        String[] symbols = new String[profile.vocabularySize];
        Iterator<String> tokens = vocabulary.keys();
        while (tokens.hasNext()) {
            String token = tokens.next();
            int index = vocabulary.getInt(token);
            require(index >= 0 && index < symbols.length && symbols[index] == null,
                    "The CTC vocabulary contains an invalid or duplicate token ID");
            symbols[index] = token;
        }
        int blankId = config.getInt("pad_token_id");
        require(blankId >= 0 && blankId < symbols.length
                        && tokenizer.getString("pad_token").equals(symbols[blankId]),
                "The CTC blank token does not match the tokenizer");
        String delimiter = tokenizer.getString("word_delimiter_token");
        require(vocabulary.has(delimiter), "The CTC vocabulary is missing its word separator");
        return new CtcDecoder(symbols, blankId, delimiter, tokenizer.getBoolean("do_lower_case"));
    }

    private static boolean matches(JSONArray array, int[] expected) throws Exception {
        if (array.length() != expected.length) {
            return false;
        }
        for (int index = 0; index < expected.length; index++) {
            if (array.getInt(index) != expected[index]) {
                return false;
            }
        }
        return true;
    }

    private static JSONObject readJson(File directory, String name) throws Exception {
        File file = requiredFile(directory, name);
        if (file.length() > 1024 * 1024) {
            throw new IllegalArgumentException("The CTC configuration is too large: " + name);
        }
        return new JSONObject(new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8));
    }

    private static File requiredFile(File directory, String name) {
        if (directory == null || !directory.isDirectory()) {
            throw new IllegalArgumentException("The selected model directory is missing");
        }
        File file = new File(directory, name);
        if (!file.isFile() || file.length() == 0) {
            throw new IllegalArgumentException("The model package is missing " + name);
        }
        return file;
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }

    private static final class Profile {
        final int inputSamples;
        final int vocabularySize;
        final int outputFrames;
        final int[] convKernels;
        final int[] convStrides;

        private Profile(int inputSamples, int vocabularySize, int[] kernels, int[] strides) {
            this.inputSamples = inputSamples;
            this.vocabularySize = vocabularySize;
            convKernels = kernels;
            convStrides = strides;
            outputFrames = CtcAudioProcessor.outputFrames(inputSamples, kernels, strides);
        }

        static Profile forId(String id) {
            if (CONFIG_CONFORMER_LARGE.equals(id)) {
                // Inspected Arm export: forward(float32[1,480000]) ->
                // float32[1,1499,32]. It has no attention-mask input.
                return new Profile(480000, 32,
                        new int[]{10, 3, 3, 3, 3, 2, 2},
                        new int[]{5, 2, 2, 2, 2, 2, 2});
            }
            throw new IllegalArgumentException("Unsupported CTC configuration: " + id);
        }
    }
}
