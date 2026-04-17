# bitnet.cpp Project Overview

`bitnet.cpp` is an official inference framework for 1-bit Large Language Models (LLMs), such as BitNet b1.58. It is built upon the `llama.cpp` framework and provides highly optimized kernels for fast and lossless inference on both CPU and GPU.

## Project Structure

- **`3rdparty/llama.cpp/`**: Core inference engine submodule.
- **`src/`**: Optimized CPU kernels and `ggml` backend extensions for BitNet.
- **`include/`**: C++ header files for the BitNet API and kernel configurations.
- **`gpu/`**: CUDA-based GEMV kernels and specialized inference scripts for NVIDIA GPUs.
- **`utils/`**: Python scripts for model conversion, benchmarking, and kernel code generation.
- **`preset_kernels/`**: Pre-tuned kernel configurations for specific model architectures.
- **`models/`**: Default directory for storing downloaded and converted GGUF models.

## Key Technologies

- **C++**: Main language for the inference engine and kernels.
- **Python**: Used for environment setup, model conversion, and automation scripts.
- **CUDA**: Powers the high-performance GPU kernels (W2A8 GEMV).
- **CMake**: Build system for the project.
- **Lookup Table (LUT)**: Methodology used for CPU kernel optimization, inspired by T-MAC.

## Building and Running

### CPU Inference (Recommended)

1.  **Environment Setup & Build**:
    Use `setup_env.py` to automate the build process, which includes generating optimized kernels and compiling the project.
    ```bash
    # Example for BitNet 2B model on CPU
    python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
    ```
    *Options:* `-q` supports `i2_s` (standard) and `tl1`/`tl2` (LUT-based optimizations for ARM/x86).

2.  **Run Inference**:
    ```bash
    python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "Your prompt here" -cnv
    ```

3.  **Run API Server**:
    Launch an HTTP server for programmatic model access.
    ```bash
    python run_inference_server.py -m <path_to_gguf_model> --port 8080
    ```

4.  **Benchmarking**:
    ```bash
    python utils/e2e_benchmark.py -m <path_to_gguf_model>
    ```

### UI Usage (Comparison Tool)

The project includes a Streamlit-based UI for comparing BitNet models against each other or against Ollama-hosted models.

1.  **Launch the UI**:
    ```bash
    streamlit run run_inference_ui.py
    ```

2.  **Features**:
    - Select and run multiple BitNet models simultaneously.
    - Compare performance and output against local Ollama models.
    - Configurable inference parameters (Threads, N-Predict, Temperature) directly from the sidebar.
    - Optimized for Mac/Local testing with concurrent execution support.

### GPU Inference

1.  **Build GPU Kernels**:
    ```bash
    cd gpu/bitnet_kernels
    bash compile.sh
    cd ..
    ```

2.  **Model Conversion & Inference**:
    Follow the instructions in `gpu/README.md` to convert `.safetensors` to the specialized format required for the GPU kernels and run `generate.py`.

## Development Conventions

- **Toolchain**: Requires `cmake >= 3.22` and `clang >= 18`.
- **Code Generation**: Optimized CPU kernels are generated dynamically during setup via `utils/codegen_tl1.py` (ARM) or `utils/codegen_tl2.py` (x86). These scripts use model-specific parameters (`BM`, `BK`, `bm`) to maximize performance.
- **Kernel Configurations**: `include/kernel_config.ini` and `include/bitnet-lut-kernels.h` are often auto-generated or copied from `preset_kernels/`.
- **Testing**: Use `utils/e2e_benchmark.py` for CPU and `gpu/test.py` for GPU kernel performance validation.
- **Architecture**: The project extends `ggml` by adding BitNet-specific tensor types and compute tasks.

## Conversion Utilities

- **`utils/convert-helper-bitnet.py`**: High-level script for converting Hugging Face models.
- **`utils/convert-hf-to-gguf-bitnet.py`**: Specialized GGUF converter for BitNet models.
- **`utils/generate-dummy-bitnet-model.py`**: Useful for benchmarking architectural layouts without downloading full models.
