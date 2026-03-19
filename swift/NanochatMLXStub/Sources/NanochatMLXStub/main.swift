import Foundation
import MLX
import MLXNN

struct WorkerRequest: Decodable {
    let promptTokens: [Int]
    let maxNewTokens: Int
    let stopTokenIds: [Int]

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case maxNewTokens = "max_new_tokens"
        case stopTokenIds = "stop_token_ids"
    }
}

struct WorkerResponse: Encodable {
    let ok: Bool
    let generatedTokenIds: [Int]
    let timing: [String: String]
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case generatedTokenIds = "generated_token_ids"
        case timing
        case error
    }
}

struct ExportSection: Decodable {
    let format: String
    let tensorCount: Int
    let safetensorsPath: String

    enum CodingKeys: String, CodingKey {
        case format
        case tensorCount = "tensor_count"
        case safetensorsPath = "safetensors_path"
    }
}

struct ConfigSection: Decodable {
    let sequenceLen: Int
    let vocabSize: Int
    let nLayer: Int
    let nHead: Int
    let nKvHead: Int
    let nEmbd: Int
    let windowPattern: String

    enum CodingKeys: String, CodingKey {
        case sequenceLen = "sequence_len"
        case vocabSize = "vocab_size"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case nKvHead = "n_kv_head"
        case nEmbd = "n_embd"
        case windowPattern = "window_pattern"
    }
}

struct TokenizerSection: Decodable {
    let sharedVocabUsed: Bool
    let bosTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case sharedVocabUsed = "shared_vocab_used"
        case bosTokenId = "bos_token_id"
    }
}

struct ExportManifest: Decodable {
    let export: ExportSection
    let config: ConfigSection
    let tokenizer: TokenizerSection
    let tensorNames: [String]

    enum CodingKeys: String, CodingKey {
        case export
        case config
        case tokenizer
        case tensorNames = "tensor_names"
    }
}

struct CLIOptions {
    let manifestURL: URL
    let promptTokens: [Int]
    let maxNewTokens: Int
    let stopTokenIds: [Int]
    let useGPU: Bool
    let serveStdin: Bool
}

enum StubError: Error, LocalizedError {
    case usage(String)
    case missingTensor(String)
    case invalidManifest(String)
    case invalidPrompt(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message):
            return message
        case .missingTensor(let name):
            return "Missing required tensor: \(name)"
        case .invalidManifest(let message):
            return message
        case .invalidPrompt(let message):
            return message
        }
    }
}

func hasValueEmbedding(_ layerIndex: Int, nLayer: Int) -> Bool {
    layerIndex % 2 == (nLayer - 1) % 2
}

func usageText() -> String {
    """
    Usage:
      swift run --package-path swift/NanochatMLXStub nanochat-mlx-stub \
        --manifest runs/mlx_exports/phase2_d4_l_mps_step20.json \
        --prompt-tokens 32759,464,1223 \
        --max-new-tokens 8

    Options:
      --manifest <path>       Path to the export sidecar JSON.
      --prompt-tokens <ids>   Comma-separated token ids. Tokenization stays Python-side for now.
      --max-new-tokens <n>    Number of greedy tokens to generate. Default: 1.
      --stop-token-ids <ids>  Optional comma-separated stop token ids.
      --device <cpu|gpu>      Execution device. Default: cpu. GPU is not wired yet in this first stub.
            --serve-stdin           Start a persistent JSON-lines worker on stdin/stdout.
    """
}

func parseCSVIntegers(_ value: String, optionName: String) throws -> [Int] {
    try value
        .split(separator: ",")
        .map { part in
            let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let parsed = Int(trimmed) else {
                throw StubError.invalidPrompt("\(optionName) must contain only integers: \(value)")
            }
            return parsed
        }
}

func parseArguments() throws -> CLIOptions {
    let args = Array(CommandLine.arguments.dropFirst())
    var manifestPath: String?
    var promptTokensValue: String?
    var stopTokenIdsValue: String?
    var device = "cpu"
    var maxNewTokens = 1
    var serveStdin = false

    var index = 0
    while index < args.count {
        let arg = args[index]
        if arg == "--serve-stdin" {
            serveStdin = true
            index += 1
            continue
        }
        guard index + 1 < args.count else {
            throw StubError.usage(usageText())
        }
        switch arg {
        case "--manifest":
            manifestPath = args[index + 1]
        case "--prompt-tokens":
            promptTokensValue = args[index + 1]
        case "--max-new-tokens":
            guard let parsed = Int(args[index + 1]), parsed >= 1 else {
                throw StubError.usage("--max-new-tokens must be a positive integer.\n\n\(usageText())")
            }
            maxNewTokens = parsed
        case "--stop-token-ids":
            stopTokenIdsValue = args[index + 1]
        case "--device":
            device = args[index + 1]
        case "--help", "-h":
            throw StubError.usage(usageText())
        default:
            throw StubError.usage("Unknown argument: \(arg)\n\n\(usageText())")
        }
        index += 2
    }

    guard let manifestPath else {
        throw StubError.usage("Missing --manifest.\n\n\(usageText())")
    }
    guard let promptTokensValue else {
        throw StubError.usage("Missing --prompt-tokens.\n\n\(usageText())")
    }

    let promptTokens = try parseCSVIntegers(promptTokensValue, optionName: "--prompt-tokens")

    guard !promptTokens.isEmpty else {
        throw StubError.invalidPrompt("Prompt token list is empty")
    }

    let stopTokenIds = try stopTokenIdsValue.map { try parseCSVIntegers($0, optionName: "--stop-token-ids") } ?? []

    let manifestURL = URL(fileURLWithPath: manifestPath)
    let normalizedDevice = device.lowercased()
    guard normalizedDevice == "gpu" || normalizedDevice == "cpu" else {
        throw StubError.usage("Unsupported --device value: \(device). Expected cpu or gpu.")
    }
    return CLIOptions(
        manifestURL: manifestURL,
        promptTokens: promptTokens,
        maxNewTokens: maxNewTokens,
        stopTokenIds: stopTokenIds,
        useGPU: normalizedDevice == "gpu",
        serveStdin: serveStdin
    )
}

func timingDict(deviceLabel: String, loadTimeMs: Double, prefillTimeMs: Double, avgDecodeMs: Double, tokensDecoded: Int) -> [String: String] {
    [
        "device": deviceLabel,
        "load": String(format: "%.1fms", loadTimeMs),
        "ttft": String(format: "%.1fms", prefillTimeMs),
        "prefill": String(format: "%.1fms", prefillTimeMs),
        "avg_decode": String(format: "%.2fms", avgDecodeMs),
        "tokens_decoded": String(tokensDecoded),
        "active_memory_gb": String(format: "%.3f", Double(Memory.activeMemory) / pow(1024.0, 3.0)),
        "peak_memory_gb": String(format: "%.3f", Double(Memory.peakMemory) / pow(1024.0, 3.0)),
        "cache_memory_gb": String(format: "%.3f", Double(Memory.cacheMemory) / pow(1024.0, 3.0)),
    ]
}

func loadManifest(_ url: URL) throws -> ExportManifest {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(ExportManifest.self, from: data)
}

func resolveCheckpointURL(manifestURL: URL, checkpointPath: String) -> URL {
    let checkpointURL = URL(fileURLWithPath: checkpointPath)
    if checkpointURL.path.hasPrefix("/") {
        return checkpointURL
    }

    let fileManager = FileManager.default
    let workingDirectoryURL = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    let cwdResolved = workingDirectoryURL.appendingPathComponent(checkpointPath)
    if fileManager.fileExists(atPath: cwdResolved.path) {
        return cwdResolved
    }
    return manifestURL.deletingLastPathComponent().appendingPathComponent(checkpointPath)
}

func requireTensor(_ name: String, in tensors: [String: MLXArray]) throws -> MLXArray {
    guard let tensor = tensors[name] else {
        throw StubError.missingTensor(name)
    }
    return tensor
}

func buildExpectedTensorNames(config: ConfigSection) -> [String] {
    var names = [
        "lm_head.weight",
        "resid_lambdas",
        "wte.weight",
        "x0_lambdas",
    ]
    for layerIndex in 0 ..< config.nLayer {
        names.append(contentsOf: [
            "blocks.\(layerIndex).attn.c_k.weight",
            "blocks.\(layerIndex).attn.c_proj.weight",
            "blocks.\(layerIndex).attn.c_q.weight",
            "blocks.\(layerIndex).attn.c_v.weight",
            "blocks.\(layerIndex).mlp.c_fc.weight",
            "blocks.\(layerIndex).mlp.c_proj.weight",
        ])
        if hasValueEmbedding(layerIndex, nLayer: config.nLayer) {
            names.append("blocks.\(layerIndex).attn.ve_gate.weight")
            names.append("value_embeds.\(layerIndex).weight")
        }
    }
    return names.sorted()
}

func validateManifest(manifest: ExportManifest, tensors: [String: MLXArray], metadata: [String: String]) throws {
    guard manifest.export.format == "nanochat-mlx-prototype" else {
        throw StubError.invalidManifest("Unsupported export format: \(manifest.export.format)")
    }
    guard metadata["format"] == "nanochat-mlx-prototype" else {
        throw StubError.invalidManifest("Safetensors metadata format mismatch: \(metadata["format"] ?? "missing")")
    }

    let expected = buildExpectedTensorNames(config: manifest.config)
    guard expected == manifest.tensorNames.sorted() else {
        throw StubError.invalidManifest("Manifest tensor names do not match the expected MLX prototype surface")
    }
    guard tensors.count == manifest.export.tensorCount else {
        throw StubError.invalidManifest("Tensor count mismatch. Manifest says \(manifest.export.tensorCount), safetensors loaded \(tensors.count)")
    }
    for name in expected {
        _ = try requireTensor(name, in: tensors)
    }
}

// ---------------------------------------------------------------------------
// MARK: - KV Cache
// ---------------------------------------------------------------------------

final class KVCache: @unchecked Sendable {
    private var keys: [MLXArray?]
    private var values: [MLXArray?]
    private(set) var length: Int
    private let maxTokens: Int

    /// Pre-allocate key/value buffers for a fixed maximum number of tokens.
    /// Avoids the growing O(n²) concatenation that building the cache step-by-step would require.
    /// Buffers are allocated lazily on first use so we can match the dtype of the activations.
    init(nLayers: Int, maxTokens: Int) {
        keys = Array(repeating: nil, count: nLayers)
        values = Array(repeating: nil, count: nLayers)
        length = 0
        self.maxTokens = maxTokens
    }

    /// Write new K, V into the pre-allocated buffer and return a view over all stored positions.
    func update(layerIndex: Int, newK: MLXArray, newV: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = newK.shape[2]
        if keys[layerIndex] == nil {
            // Lazy allocation: use dtype of the first activation tensor.
            keys[layerIndex] = zeros(
                [newK.shape[0], newK.shape[1], maxTokens, newK.shape[3]], dtype: newK.dtype)
            values[layerIndex] = zeros(
                [newV.shape[0], newV.shape[1], maxTokens, newV.shape[3]], dtype: newV.dtype)
        }
        var k = keys[layerIndex]!
        var v = values[layerIndex]!
        k[0..., 0..., length..<(length + seqLen), 0...] = newK
        v[0..., 0..., length..<(length + seqLen), 0...] = newV
        keys[layerIndex] = k
        values[layerIndex] = v
        return (
            k[0..., 0..., 0..<(length + seqLen), 0...],
            v[0..., 0..., 0..<(length + seqLen), 0...]
        )
    }

    func advance(by count: Int) { length += count }
}

// ---------------------------------------------------------------------------
// MARK: - Helpers
// ---------------------------------------------------------------------------

func rmsNormNoWeight(_ x: MLXArray, width: Int) -> MLXArray {
    let weight = ones([width], dtype: x.dtype)
    return rmsNorm(x, weight: weight, eps: 1e-5)
}

func applyLinear(_ x: MLXArray, weight: MLXArray) -> MLXArray {
    tensordot(x, weight, axes: ([-1], [1]))
}

func finalLogitsStep(_ logits: MLXArray, sequenceLength: Int) -> MLXArray {
    logits.split(indices: [sequenceLength - 1], axis: 1)[1]
}

func greedyNextTokenId(_ logits: MLXArray) throws -> Int {
    let nextToken = argMax(logits, axis: -1)
    try checkedEval(nextToken)
    return Int(nextToken.item(UInt32.self))
}

struct AttentionLayer {
    let cQ: MLXArray
    let cK: MLXArray
    let cV: MLXArray
    let cProj: MLXArray
    let rope: RoPE
    let veGate: MLXArray?
    let nHead: Int
    let nKvHead: Int
    let nEmbd: Int
    let headDim: Int

    func callAsFunction(_ x: MLXArray, ve: MLXArray?, cache: KVCache?, layerIndex: Int) -> MLXArray {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]
        let offset = cache?.length ?? 0

        var q = applyLinear(x, weight: cQ).reshaped(batchSize, seqLen, nHead, headDim).transposed(0, 2, 1, 3)
        var k = applyLinear(x, weight: cK).reshaped(batchSize, seqLen, nKvHead, headDim).transposed(0, 2, 1, 3)
        var v = applyLinear(x, weight: cV).reshaped(batchSize, seqLen, nKvHead, headDim).transposed(0, 2, 1, 3)

        if let ve, let veGate {
            let gateInput = x.split(indices: [12], axis: -1)[0]
            var gate = 3.0 * sigmoid(applyLinear(gateInput, weight: veGate))
            gate = gate.transposed(0, 2, 1).reshaped(batchSize, nKvHead, seqLen, 1)
            let veReshaped = ve.reshaped(batchSize, seqLen, nKvHead, headDim).transposed(0, 2, 1, 3)
            v = v + gate * veReshaped
        }

        // Apply RoPE with position offset (positions start at `offset` for KV-cache decode)
        q = rmsNormNoWeight(rope(q, offset: offset), width: headDim) * 1.15
        k = rmsNormNoWeight(rope(k, offset: offset), width: headDim) * 1.15

        // Update KV-cache (K, V are stored post-RoPE for K)
        if let cache {
            let (fullK, fullV) = cache.update(layerIndex: layerIndex, newK: k, newV: v)
            k = fullK
            v = fullV
        }

        let totalKLen = k.shape[2]

        // Causal mask only needed for prefill (seqLen > 1);
        // during decode (seqLen == 1) the single query attends to all cached positions.
        let mask: MLXArray?
        if seqLen > 1 {
            let causal = 1.0 - tri(seqLen, m: totalKLen, k: 0, dtype: .float32)
            mask = (causal * -1.0e9).reshaped(1, 1, seqLen, totalKLen)
        } else {
            mask = nil
        }

        let attended = scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0 / sqrt(Float(headDim)),
            mask: mask
        )
        let merged = attended.reshaped(batchSize, seqLen, nEmbd)
        return applyLinear(merged, weight: cProj)
    }
}

struct MLPBlock {
    let cFc: MLXArray
    let cProj: MLXArray

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = relu(applyLinear(x, weight: cFc))
        return applyLinear(square(hidden), weight: cProj)
    }
}

struct TransformerBlock {
    let attention: AttentionLayer
    let mlp: MLPBlock
    let width: Int

    func callAsFunction(_ x: MLXArray, ve: MLXArray?, cache: KVCache?, layerIndex: Int) -> MLXArray {
        let withAttention = x + attention(rmsNormNoWeight(x, width: width), ve: ve, cache: cache, layerIndex: layerIndex)
        return withAttention + mlp(rmsNormNoWeight(withAttention, width: width))
    }
}

struct NanochatPrototype {
    let config: ConfigSection
    let tokenEmbedding: Embedding
    let lmHead: Embedding
    let residLambdas: MLXArray
    let x0Lambdas: MLXArray
    let valueEmbeds: [Embedding?]
    let blocks: [TransformerBlock]

    init(config: ConfigSection, tensors: [String: MLXArray]) throws {
        self.config = config
        self.tokenEmbedding = Embedding(weight: try requireTensor("wte.weight", in: tensors))
        self.lmHead = Embedding(weight: try requireTensor("lm_head.weight", in: tensors))
        self.residLambdas = try requireTensor("resid_lambdas", in: tensors)
        self.x0Lambdas = try requireTensor("x0_lambdas", in: tensors)

        var valueEmbeds = [Embedding?]()
        var blocks = [TransformerBlock]()
        let headDim = config.nEmbd / config.nHead

        for layerIndex in 0 ..< config.nLayer {
            let rope = RoPE(dimensions: headDim, traditional: false, base: 100000, scale: 1.0)
            let veGateName = "blocks.\(layerIndex).attn.ve_gate.weight"
            let valueEmbedName = "value_embeds.\(layerIndex).weight"
            let valueEmbed = tensors[valueEmbedName].map { Embedding(weight: $0) }
            valueEmbeds.append(valueEmbed)

            let attention = AttentionLayer(
                cQ: try requireTensor("blocks.\(layerIndex).attn.c_q.weight", in: tensors),
                cK: try requireTensor("blocks.\(layerIndex).attn.c_k.weight", in: tensors),
                cV: try requireTensor("blocks.\(layerIndex).attn.c_v.weight", in: tensors),
                cProj: try requireTensor("blocks.\(layerIndex).attn.c_proj.weight", in: tensors),
                rope: rope,
                veGate: tensors[veGateName],
                nHead: config.nHead,
                nKvHead: config.nKvHead,
                nEmbd: config.nEmbd,
                headDim: headDim
            )
            let mlp = MLPBlock(
                cFc: try requireTensor("blocks.\(layerIndex).mlp.c_fc.weight", in: tensors),
                cProj: try requireTensor("blocks.\(layerIndex).mlp.c_proj.weight", in: tensors)
            )
            blocks.append(TransformerBlock(attention: attention, mlp: mlp, width: config.nEmbd))
        }

        self.valueEmbeds = valueEmbeds
        self.blocks = blocks
    }

    func callAsFunction(_ tokenIds: MLXArray, cache: KVCache? = nil) -> MLXArray {
        var x = tokenEmbedding(tokenIds)
        x = rmsNormNoWeight(x, width: config.nEmbd)
        let x0 = x

        for layerIndex in 0 ..< config.nLayer {
            let resid = residLambdas[layerIndex]
            let x0Lambda = x0Lambdas[layerIndex]
            x = resid * x + x0Lambda * x0
            let ve = valueEmbeds[layerIndex]?(tokenIds)
            x = blocks[layerIndex](x, ve: ve, cache: cache, layerIndex: layerIndex)
        }

        x = rmsNormNoWeight(x, width: config.nEmbd)
        let logits = lmHead.asLinear(x).asType(.float32)
        return 15.0 * tanh(logits / 15.0)
    }
}

func generateTokenIds(
    model: NanochatPrototype,
    promptTokens: [Int],
    maxNewTokens: Int,
    stopTokenIds: [Int],
    loadTimeMs: Double,
    deviceLabel: String
) throws -> (generatedTokenIds: [Int], logitsShape: [Int], timing: [String: String]) {
    Memory.peakMemory = 0
    let cache = KVCache(nLayers: model.config.nLayer, maxTokens: promptTokens.count + maxNewTokens)
    let prefillStart = CFAbsoluteTimeGetCurrent()
    let promptArray = MLXArray(promptTokens.map(Int32.init), [1, promptTokens.count])
    let prefillLogits = model(promptArray, cache: cache)
    try checkedEval(prefillLogits)
    cache.advance(by: promptTokens.count)
    let prefillTimeMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000.0

    let finalLogitsShape = prefillLogits.shape
    let firstTokenId = try greedyNextTokenId(finalLogitsStep(prefillLogits, sequenceLength: promptTokens.count))

    var generatedTokenIds = [firstTokenId]
    var allTokenIds = promptTokens + [firstTokenId]
    var decodeTimesMs: [Double] = []

    if maxNewTokens > 1 && !stopTokenIds.contains(firstTokenId) {
        for _ in 1 ..< maxNewTokens {
            let decodeStart = CFAbsoluteTimeGetCurrent()
            let newToken = MLXArray([Int32(allTokenIds.last!)], [1, 1])
            let logits = model(newToken, cache: cache)
            try checkedEval(logits)
            cache.advance(by: 1)
            let nextTokenId = try greedyNextTokenId(logits)
            let decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000.0
            decodeTimesMs.append(decodeMs)

            generatedTokenIds.append(nextTokenId)
            allTokenIds.append(nextTokenId)
            if stopTokenIds.contains(nextTokenId) {
                break
            }
        }
    }

    let avgDecodeMs = decodeTimesMs.isEmpty ? 0.0 : decodeTimesMs.reduce(0.0, +) / Double(decodeTimesMs.count)
    let tokensDecoded = decodeTimesMs.count
    return (
        generatedTokenIds,
        finalLogitsShape,
        timingDict(
            deviceLabel: deviceLabel,
            loadTimeMs: loadTimeMs,
            prefillTimeMs: prefillTimeMs,
            avgDecodeMs: avgDecodeMs,
            tokensDecoded: tokensDecoded
        )
    )
}

func runWorkerLoop(model: NanochatPrototype, loadTimeMs: Double, deviceLabel: String) throws {
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let readyData = try JSONSerialization.data(withJSONObject: ["status": "ready", "device": deviceLabel])
    FileHandle.standardOutput.write(readyData)
    FileHandle.standardOutput.write(Data("\n".utf8))

    while let line = readLine() {
        if line.isEmpty {
            continue
        }
        do {
            let request = try decoder.decode(WorkerRequest.self, from: Data(line.utf8))
            let result = try generateTokenIds(
                model: model,
                promptTokens: request.promptTokens,
                maxNewTokens: request.maxNewTokens,
                stopTokenIds: request.stopTokenIds,
                loadTimeMs: loadTimeMs,
                deviceLabel: deviceLabel
            )
            let response = WorkerResponse(ok: true, generatedTokenIds: result.generatedTokenIds, timing: result.timing, error: nil)
            FileHandle.standardOutput.write(try encoder.encode(response))
            FileHandle.standardOutput.write(Data("\n".utf8))
            // Release cached GPU buffers from this request so they don't
            // inflate memory pressure for the next one.
            Memory.clearCache()
        } catch {
            let response = WorkerResponse(ok: false, generatedTokenIds: [], timing: [:], error: error.localizedDescription)
            FileHandle.standardOutput.write(try encoder.encode(response))
            FileHandle.standardOutput.write(Data("\n".utf8))
            Memory.clearCache()
        }
    }
}

func main() throws {
    let options = try parseArguments()
    let manifest = try loadManifest(options.manifestURL)
    let checkpointURL = resolveCheckpointURL(manifestURL: options.manifestURL, checkpointPath: manifest.export.safetensorsPath)

    // --- Load checkpoint (always on CPU; file I/O is a CPU operation) ---
    let loadStart = CFAbsoluteTimeGetCurrent()
    let checkpointData = try Data(contentsOf: checkpointURL)
    let (tensors, metadata) = try loadArraysAndMetadata(data: checkpointData, stream: .cpu)
    try validateManifest(manifest: manifest, tensors: tensors, metadata: metadata)
    let model = try NanochatPrototype(config: manifest.config, tensors: tensors)
    let loadTimeMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000.0

    // --- Compute on the requested device ---
    let computeDevice = options.useGPU ? Device(.gpu) : Device(.cpu)
    try Device.withDefaultDevice(computeDevice) {
        let deviceLabel = options.useGPU ? "gpu" : "cpu"
        if options.serveStdin {
            try runWorkerLoop(model: model, loadTimeMs: loadTimeMs, deviceLabel: deviceLabel)
        } else {
            let result = try generateTokenIds(
                model: model,
                promptTokens: options.promptTokens,
                maxNewTokens: options.maxNewTokens,
                stopTokenIds: options.stopTokenIds,
                loadTimeMs: loadTimeMs,
                deviceLabel: deviceLabel
            )
            print("Loaded export: \(checkpointURL.path)")
            print("Prompt token count: \(options.promptTokens.count)")
            print("Max new tokens requested: \(options.maxNewTokens)")
            print("Logits shape: \(result.logitsShape)")
            print("Generated token ids: \(result.generatedTokenIds.map(String.init).joined(separator: ","))")
            print("Timing: device=\(result.timing["device"] ?? deviceLabel) load=\(result.timing["load"] ?? "0.0ms") prefill=\(result.timing["prefill"] ?? "0.0ms") avg_decode=\(result.timing["avg_decode"] ?? "0.00ms") tokens_decoded=\(result.timing["tokens_decoded"] ?? "0")")
        }
    }
}

do {
    try main()
} catch {
    fputs("\(error.localizedDescription)\n", stderr)
    exit(1)
}