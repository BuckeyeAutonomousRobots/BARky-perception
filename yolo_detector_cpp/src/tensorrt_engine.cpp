#include "yolo_detector_cpp/tensorrt_engine.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

Logger gLogger;

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

size_t elements(const nvinfer1::Dims& dims) {
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Dynamic or invalid tensor shape");
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

} // namespace

TensorRTEngine::TensorRTEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open engine: " + path);
    }

    std::vector<char> data(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) {
        throw std::runtime_error("Cannot create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(
        data.data(), data.size()
    ));
    if (!engine_) {
        throw std::runtime_error("Cannot deserialize TensorRT engine");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Cannot create execution context");
    }

    if (engine_->getNbIOTensors() != 2) {
        throw std::runtime_error("Expected one input and one output tensor");
    }

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);

        if (engine_->getTensorDataType(name) !=
            nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Expected FP32 I/O tensors");
        }

        const auto mode = engine_->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            input_elements_ = elements(engine_->getTensorShape(name));
        } else {
            output_name_ = name;
            output_elements_ = elements(engine_->getTensorShape(name));
        }
    }

    if (input_elements_ != 3 * 640 * 640 ||
        output_elements_ != 300 * 6) {
        throw std::runtime_error("Unexpected YOLO tensor dimensions");
    }
    // GPU allocation
    try {
        // create cuda stream using null pointer in tensorrt hpp 
        checkCuda(cudaStreamCreate(&stream_));
        
        // allocates vram on graphics card 
        // malloc creates an empty container in VRam that is the
        // size of the input tensor shape and return back the raw pointer to the empty block 
        // input buffer and output buffer but the GPU doesn't know what it is yet 
        checkCuda(cudaMalloc(
            &input_buffer_, input_elements_ * sizeof(float)
        ));
        checkCuda(cudaMalloc(
            &output_buffer_, output_elements_ * sizeof(float)
        ));
        // binds the input/output name to the input/output pointer address in GPU memory 
        if (!context_->setTensorAddress(
                input_name_.c_str(), input_buffer_) ||
            !context_->setTensorAddress(
                output_name_.c_str(), output_buffer_)) {
            throw std::runtime_error("Failed to bind TensorRT tensors");
        }
        // if there is any exception ...
    } catch (...) {
        // frees reserved VRAM slots and closes stream queue 
        // and resets pointers to blank state
        if (input_buffer_) cudaFree(input_buffer_);
        if (output_buffer_) cudaFree(output_buffer_);
        if (stream_) cudaStreamDestroy(stream_);
        input_buffer_ = nullptr;
        output_buffer_ = nullptr;
        stream_ = nullptr;
        throw;
    }

    std::cout << "TensorRT engine loaded successfully\n";
}
// when TensorRTEngine gies out of scope or is deleted 
TensorRTEngine::~TensorRTEngine() {
    if (stream_) cudaStreamSynchronize(stream_);
    if (input_buffer_) cudaFree(input_buffer_);
    if (output_buffer_) cudaFree(output_buffer_);
    if (stream_) cudaStreamDestroy(stream_);
}
// inference 
bool TensorRTEngine::infer(
    const float* input,
    std::vector<float>& output
) {
    // Safety check to ensure we didn't get passed a null pointer
    if (input == nullptr) {
        return false;
    }

    // Resizes CPU output vector so that it is large enough to receive results from GPU
    output.resize(output_elements_);

    // Copies raw input data from CPU RAM pointer directly to GPU VRAM 
    checkCuda(cudaMemcpyAsync(
        input_buffer_,
        input,                               // Passed raw CPU pointer directly
        input_elements_ * sizeof(float),
        cudaMemcpyHostToDevice,
        stream_
    ));

    // Uses TensorRT execution API to read from input buffer and write
    // raw predictions into output buffer asynchronously
    if (!context_->enqueueV3(stream_)) {
        return false;
    }

    // Pulls results back from GPU into output vector in CPU 
    checkCuda(cudaMemcpyAsync(
        output.data(), // destination: CPU RAM pointer y
        output_buffer_, // Source: GPU VRAM Pointer 
        output_elements_ * sizeof(float),
        cudaMemcpyDeviceToHost, // Direction 
        stream_
    ));

    // Blocks CPU until everything in the stream is completely done
    checkCuda(cudaStreamSynchronize(stream_));

    return true;
}