# Nano-vLLM (edited by sujkilla 06/30/2025)

A lightweight vLLM implementation built from scratch with support for multiple model architectures.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 🦙 **Multi-architecture support** - Supports both Qwen3 and Llama 3.1 models
* 📊 **Built-in benchmarking** - Comprehensive performance comparison tools

## Supported Models

### Qwen3 Models
- Qwen3-0.6B, 1.8B, 4B, 8B, 14B, 32B
- Original focus of the nano-vLLM implementation

### Llama 3.1 Models (sujkilla 06/30/2025)
- Llama-3.1-8B, 70B (8B focus for this implementation)
- Features grouped query attention (GQA)
- No query/key normalization (vs Qwen3)
- Optimized for 128K context length

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Downloads

### Qwen3 Model
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### Llama 3.1 8B Model (sujkilla 06/30/2025)
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/huggingface/Llama-3.1-8B-Instruct/ \
  --local-dir-use-symlinks False
```

> **Note**: You'll need to accept Meta's license agreement on Hugging Face to download Llama models.

## Quick Start

### Basic Usage (Model-agnostic)

The API mirrors vLLM's interface with minor differences in the `LLM.generate` method. Nano-vLLM automatically detects the model architecture:

```python
from nanovllm import LLM, SamplingParams

# Works with both Qwen3 and Llama models
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

### Example Scripts

- **`example.py`** - Original Qwen3 example
- **`example_llama.py`** - Llama 3.1 8B example with chat templates
- **`bench_comparison.py`** - Compare performance between models

### Qwen3 Example
```bash
python example.py
```

### Llama 3.1 Example (sujkilla 06/30/2025)
```bash
python example_llama.py
```

## Benchmarking

Multiple benchmark scripts are available for performance testing:

- **`bench.py`** - Original Qwen3 benchmark
- **`bench_llama.py`** - Llama 3.1 8B benchmark (sujkilla 06/30/2025)
- **`bench_comparison.py`** - Side-by-side model comparison

### Run Comparison Benchmark
```bash
python bench_comparison.py
```

### Original Qwen3 Performance Results

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

### Model Architecture Comparison (sujkilla 06/30/2025)

| Feature | Qwen3 | Llama 3.1 |
|---------|-------|-----------|
| Query/Key Normalization | ✅ | ❌ |
| Grouped Query Attention | ✅ | ✅ |
| RoPE Positional Encoding | ✅ | ✅ |
| SiLU Activation | ✅ | ✅ |
| Context Length | 32K | 128K |
| Architecture Focus | Efficiency | Scale |


## Implementation Details (sujkilla 06/30/2025)

### Llama 3.1 Architecture Implementation

This fork extends the original nano-vLLM (focused on Qwen3) to support Llama 3.1 8B models. Key architectural differences implemented:

#### **Attention Mechanism**
- **Removed Query/Key Normalization**: Unlike Qwen3, Llama 3.1 does not use RMSNorm on query and key projections
- **Grouped Query Attention**: Implemented GQA with 32 query heads and 8 key-value heads (4:1 ratio)
- **RoPE Configuration**: Updated to use Llama's rope_theta=500000 and scaling parameters

#### **Model Configuration**
- **LlamaConfig Integration**: Uses `transformers.LlamaConfig` instead of `Qwen3Config`
- **Architecture Detection**: Automatic model type detection based on `model_type` field
- **Context Length**: Support for 128K context length (vs 32K in Qwen3)

#### **Files Added/Modified**

**New Files:**
- `nanovllm/models/llama3.py` - Complete Llama 3.1 model implementation
- `example_llama.py` - Llama-specific example with chat templates
- `bench_llama.py` - Llama benchmarking script
- `bench_comparison.py` - Side-by-side model comparison

**Modified Files:**
- `nanovllm/engine/model_runner.py` - Added model architecture detection
- `README.md` - Updated documentation for multi-model support

### Performance Characteristics

The implementation maintains the same optimization features as the original:
- **Prefix Caching**: Efficient KV cache management
- **Tensor Parallelism**: Multi-GPU support 
- **CUDA Graphs**: Optimized decode paths
- **FlashAttention**: Memory-efficient attention computation

### Usage Notes

1. **Model Detection**: The system automatically detects model architecture from the HuggingFace config
2. **Memory Requirements**: Llama 3.1 8B requires more memory than Qwen3-0.6B
3. **Chat Templates**: Llama models use different chat templates than Qwen models
4. **Tokenizer**: Each model family uses its own tokenizer vocabulary

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
