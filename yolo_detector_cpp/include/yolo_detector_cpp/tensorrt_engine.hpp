#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

class TensorRTEngine
{
public:
    explicit TensorRTEngine(const std::string& engine_path);

    ~TensorRTEngine();

    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    bool infer(
        const float* input,
        std::vector<float>& output
    );

private:
    struct TRTDeleter {
        template<typename T>
        void operator()(T* ptr) const {
            delete ptr;
        }
    };

    std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> context_;

    void* input_buffer_ = nullptr;
    void* output_buffer_ = nullptr;

    cudaStream_t stream_ = nullptr;

    std::string input_name_;
    std::string output_name_;

    size_t input_elements_ = 0;
    size_t output_elements_ = 0;
};