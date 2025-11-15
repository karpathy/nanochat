/**
 * ONNX Runtime Inference Example for nanochat
 * 
 * This example demonstrates how to load and run inference with a nanochat
 * model exported to ONNX format using ONNX Runtime C++ API.
 * 
 * Build:
 *   mkdir build && cd build
 *   cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ..
 *   cmake --build . --config Release
 * 
 * Run:
 *   ./onnx_inference ../model.onnx
 */

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

class NanoChatONNXInference {
public:
    NanoChatONNXInference(const std::string& model_path, bool use_cuda = false) {
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NanoChat");
        
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Add CUDA provider if requested
        if (use_cuda) {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "Using CUDA execution provider" << std::endl;
        } else {
            std::cout << "Using CPU execution provider" << std::endl;
        }
        
        // Load the model
        std::cout << "Loading ONNX model from: " << model_path << std::endl;
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.reserve(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.get());
        }
        
        // Output info
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.get());
        }
        
        std::cout << "✓ Model loaded successfully" << std::endl;
        std::cout << "  Inputs: ";
        for (const auto& name : input_names_) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        std::cout << "  Outputs: ";
        for (const auto& name : output_names_) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
    }
    
    /**
     * Run inference on a sequence of token IDs.
     * 
     * @param input_ids Vector of token IDs
     * @return Logits vector of shape [batch_size * seq_len * vocab_size]
     */
    std::vector<float> forward(const std::vector<int64_t>& input_ids, 
                               int64_t& batch_size, 
                               int64_t& seq_len, 
                               int64_t& vocab_size) {
        // Prepare input tensor
        batch_size = 1;
        seq_len = input_ids.size();
        
        std::vector<int64_t> input_shape = {batch_size, seq_len};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            const_cast<int64_t*>(input_ids.data()),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Prepare input names as const char*
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            &input_tensor,
            1,
            output_names_cstr.data(),
            output_names_cstr.size()
        );
        
        // Get output tensor
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        vocab_size = output_shape[2];
        size_t output_size = batch_size * seq_len * vocab_size;
        
        std::vector<float> logits(output_data, output_data + output_size);
        
        return logits;
    }
    
    /**
     * Sample next token from logits using greedy decoding.
     */
    int64_t sample_greedy(const std::vector<float>& logits, 
                         int64_t seq_len, 
                         int64_t vocab_size) {
        // Get logits for last position
        size_t last_pos_offset = (seq_len - 1) * vocab_size;
        
        // Find argmax
        auto max_it = std::max_element(
            logits.begin() + last_pos_offset,
            logits.begin() + last_pos_offset + vocab_size
        );
        
        return std::distance(logits.begin() + last_pos_offset, max_it);
    }
    
    /**
     * Sample next token with temperature and top-k sampling.
     */
    int64_t sample(const std::vector<float>& logits,
                  int64_t seq_len,
                  int64_t vocab_size,
                  float temperature = 1.0,
                  int top_k = 0) {
        // Get logits for last position
        size_t last_pos_offset = (seq_len - 1) * vocab_size;
        std::vector<float> last_logits(
            logits.begin() + last_pos_offset,
            logits.begin() + last_pos_offset + vocab_size
        );
        
        // Greedy if temperature is 0
        if (temperature <= 0.0f) {
            auto max_it = std::max_element(last_logits.begin(), last_logits.end());
            return std::distance(last_logits.begin(), max_it);
        }
        
        // Apply temperature
        for (auto& logit : last_logits) {
            logit /= temperature;
        }
        
        // Apply top-k filtering
        if (top_k > 0 && top_k < vocab_size) {
            // Get top-k indices
            std::vector<size_t> indices(vocab_size);
            std::iota(indices.begin(), indices.end(), 0);
            
            std::partial_sort(
                indices.begin(),
                indices.begin() + top_k,
                indices.end(),
                [&last_logits](size_t i1, size_t i2) {
                    return last_logits[i1] > last_logits[i2];
                }
            );
            
            float threshold = last_logits[indices[top_k - 1]];
            
            // Mask out non-top-k values
            for (size_t i = 0; i < vocab_size; ++i) {
                if (last_logits[i] < threshold) {
                    last_logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }
        
        // Compute softmax
        float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
        std::vector<float> probs(vocab_size);
        float sum = 0.0f;
        
        for (size_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(last_logits[i] - max_logit);
            sum += probs[i];
        }
        
        for (auto& p : probs) {
            p /= sum;
        }
        
        // Sample from distribution
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        
        return dist(gen);
    }
    
    /**
     * Generate tokens autoregressively.
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
            int64_t batch_size, seq_len, vocab_size;
            auto logits = forward(generated_ids, batch_size, seq_len, vocab_size);
            
            // Sample next token
            auto next_token = sample(logits, seq_len, vocab_size, temperature, top_k);
            
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
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [use_cuda]" << std::endl;
        std::cerr << "Example: " << argv[0] << " model.onnx 1" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    bool use_cuda = (argc > 2 && std::string(argv[2]) == "1");
    
    try {
        // Load model
        NanoChatONNXInference model(model_path, use_cuda);
        
        // Example prompt (replace with actual tokenized text)
        std::vector<int64_t> prompt_ids = {1, 464, 11742, 15150, 315, 3090, 374};
        
        std::cout << "\nPrompt token IDs: ";
        for (auto id : prompt_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Run single forward pass
        std::cout << "\n--- Single Forward Pass ---" << std::endl;
        int64_t batch_size, seq_len, vocab_size;
        auto logits = model.forward(prompt_ids, batch_size, seq_len, vocab_size);
        std::cout << "Output shape: [" << batch_size << ", " 
                  << seq_len << ", " << vocab_size << "]" << std::endl;
        
        // Sample next token
        auto next_token = model.sample_greedy(logits, seq_len, vocab_size);
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
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
