/**
 * LibTorch (TorchScript) Inference Example for nanochat
 * 
 * This example demonstrates how to load and run inference with a nanochat
 * model exported to TorchScript format using LibTorch C++ API.
 * 
 * Build:
 *   mkdir build && cd build
 *   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
 *   cmake --build . --config Release
 * 
 * Run:
 *   ./libtorch_inference ../model.pt
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

class NanoChatInference {
public:
    NanoChatInference(const std::string& model_path, torch::Device device = torch::kCPU)
        : device_(device) {
        try {
            // Load the TorchScript model
            std::cout << "Loading model from: " << model_path << std::endl;
            module_ = torch::jit::load(model_path);
            module_.to(device_);
            module_.eval();
            std::cout << "✓ Model loaded successfully" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }
    
    /**
     * Run inference on a sequence of token IDs.
     * 
     * @param input_ids Vector of token IDs (shape: [seq_len])
     * @return Logits tensor of shape [1, seq_len, vocab_size]
     */
    torch::Tensor forward(const std::vector<int64_t>& input_ids) {
        // Convert input to tensor
        auto options = torch::TensorOptions()
            .dtype(torch::kLong)
            .device(device_);
        
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<int64_t*>(input_ids.data()),
            {1, static_cast<int64_t>(input_ids.size())},
            torch::kLong
        ).to(device_);
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        torch::NoGradGuard no_grad;
        auto output = module_.forward(inputs).toTensor();
        
        return output;
    }
    
    /**
     * Sample next token from logits using greedy decoding.
     * 
     * @param logits Logits tensor of shape [1, seq_len, vocab_size]
     * @return Next token ID
     */
    int64_t sample_greedy(const torch::Tensor& logits) {
        // Get logits for last position
        auto last_logits = logits.index({0, -1, torch::indexing::Slice()});
        
        // Greedy sampling: argmax
        auto next_token = last_logits.argmax().item<int64_t>();
        
        return next_token;
    }
    
    /**
     * Sample next token with temperature and top-k sampling.
     * 
     * @param logits Logits tensor of shape [1, seq_len, vocab_size]
     * @param temperature Temperature for sampling (0.0 = greedy)
     * @param top_k Top-k filtering (0 = no filtering)
     * @return Next token ID
     */
    int64_t sample(const torch::Tensor& logits, float temperature = 1.0, int top_k = 0) {
        // Get logits for last position
        auto last_logits = logits.index({0, -1, torch::indexing::Slice()}).clone();
        
        // Greedy decoding if temperature is 0
        if (temperature <= 0.0f) {
            return last_logits.argmax().item<int64_t>();
        }
        
        // Apply temperature
        last_logits = last_logits / temperature;
        
        // Apply top-k filtering
        if (top_k > 0) {
            auto vocab_size = last_logits.size(0);
            auto k = std::min(top_k, static_cast<int>(vocab_size));
            
            auto topk_result = torch::topk(last_logits, k);
            auto topk_values = std::get<0>(topk_result);
            auto topk_indices = std::get<1>(topk_result);
            
            // Set all non-top-k values to -inf
            auto threshold = topk_values[-1].item<float>();
            last_logits = torch::where(
                last_logits < threshold,
                torch::full_like(last_logits, -std::numeric_limits<float>::infinity()),
                last_logits
            );
        }
        
        // Apply softmax to get probabilities
        auto probs = torch::softmax(last_logits, /*dim=*/0);
        
        // Sample from the distribution
        auto next_token = torch::multinomial(probs, /*num_samples=*/1).item<int64_t>();
        
        return next_token;
    }
    
    /**
     * Generate tokens autoregressively.
     * 
     * @param prompt_ids Initial prompt token IDs
     * @param max_tokens Maximum number of tokens to generate
     * @param temperature Temperature for sampling
     * @param top_k Top-k filtering
     * @return Generated token IDs (including prompt)
     */
    std::vector<int64_t> generate(
        const std::vector<int64_t>& prompt_ids,
        int max_tokens = 100,
        float temperature = 1.0,
        int top_k = 50
    ) {
        std::vector<int64_t> generated_ids = prompt_ids;
        
        std::cout << "Generating " << max_tokens << " tokens..." << std::endl;
        
        for (int i = 0; i < max_tokens; ++i) {
            // Forward pass
            auto logits = forward(generated_ids);
            
            // Sample next token
            auto next_token = sample(logits, temperature, top_k);
            
            // Append to sequence
            generated_ids.push_back(next_token);
            
            // Print progress
            if ((i + 1) % 10 == 0) {
                std::cout << "  Generated " << (i + 1) << "/" << max_tokens << " tokens" << std::endl;
            }
        }
        
        return generated_ids;
    }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
};


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [use_cuda]" << std::endl;
        std::cerr << "Example: " << argv[0] << " model.pt 1" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    bool use_cuda = (argc > 2 && std::string(argv[2]) == "1");
    
    // Setup device
    torch::Device device = torch::kCPU;
    if (use_cuda && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA device" << std::endl;
    } else {
        std::cout << "Using CPU device" << std::endl;
    }
    
    try {
        // Load model
        NanoChatInference model(model_path, device);
        
        // Example prompt (you would normally get these from a tokenizer)
        // These are just example token IDs - replace with actual tokenized text
        std::vector<int64_t> prompt_ids = {1, 464, 11742, 15150, 315, 3090, 374};
        // Corresponds roughly to: "The chemical formula of water is"
        
        std::cout << "\nPrompt token IDs: ";
        for (auto id : prompt_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Run single forward pass
        std::cout << "\n--- Single Forward Pass ---" << std::endl;
        auto logits = model.forward(prompt_ids);
        std::cout << "Output shape: [" << logits.size(0) << ", " 
                  << logits.size(1) << ", " << logits.size(2) << "]" << std::endl;
        
        // Sample next token
        auto next_token = model.sample_greedy(logits);
        std::cout << "Next token (greedy): " << next_token << std::endl;
        
        // Generate sequence
        std::cout << "\n--- Autoregressive Generation ---" << std::endl;
        auto generated_ids = model.generate(
            prompt_ids,
            /*max_tokens=*/20,
            /*temperature=*/0.8,
            /*top_k=*/50
        );
        
        std::cout << "\nGenerated token IDs: ";
        for (auto id : generated_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\n✓ Inference completed successfully!" << std::endl;
        std::cout << "\nNote: To decode tokens to text, you need to implement" << std::endl;
        std::cout << "      a tokenizer in C++ or use the Python tokenizer." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
