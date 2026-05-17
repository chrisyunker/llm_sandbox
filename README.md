# LLM Sandbox

A CLI tool for running local LLM inference experiments across three backend implementations: Apple MLX, llama.cpp, and PyTorch.

## Three Backends: GPU Stack Comparison

### 1. MLX (`mlx_harness.py`)
**GPU Interface:** Apple Metal (via MLX framework)

- `mlx-lm` sits on top of Apple's [MLX](https://github.com/ml-explore/mlx) framework, which compiles array operations down to **Metal shaders** on Apple Silicon.
- The Metal GPU is accessed through `mlx.core` — `mx.get_active_memory()` / `mx.get_peak_memory()` query the Metal heap directly.
- MLX uses **unified memory architecture** (UMA): CPU and GPU share the same physical RAM, so there's no PCIe data transfer. Tensors live in shared memory and the Metal GPU operates on them in-place.
- Model weights stay as MLX arrays (`mx.array`); quantized layers use `nn.QuantizedLinear` which dispatches optimized Metal kernels automatically.
- **Stack:** `mlx-lm` → `mlx.core` → Metal Performance Shaders (MPS) → Apple Silicon GPU

---

### 2. llama.cpp (`llama_harness.py`)
**GPU Interface:** Metal (macOS) or CUDA (Linux/Windows), via `llama-cpp-python`

- `llama-cpp-python` is a Python binding over [llama.cpp](https://github.com/ggerganov/llama.cpp), a C++ inference engine.
- GPU offload is controlled by `n_gpu_layers` — this parameter specifies how many transformer layers are pushed to the GPU; the rest run on CPU. This is explicit, layer-level control.
- Models must be in **GGUF format** with baked-in quantization (e.g., `*Q8_0.gguf`). The quantized kernels are hand-written in C++ and dispatched to Metal or CUDA from within llama.cpp's C++ core.
- The Python layer (`llama-cpp-python`) is a thin `ctypes`/`cffi` wrapper — Python calls pass through to compiled C++ with minimal overhead.
- **Stack:** `llama-cpp-python` (Python bindings) → `llama.cpp` (C++) → Metal/CUDA kernels → GPU

---

### 3. PyTorch (`torch_harness.py`)
**GPU Interface:** CUDA (NVIDIA) or Metal (via MPS backend)

- Uses Hugging Face `transformers` on top of PyTorch. Device selection is dynamic: `get_device()` returns `"mps"`, `"cuda"`, or `"cpu"`.
- For **CUDA**, PyTorch dispatches through **cuDNN** and **cuBLAS** for matmuls/attention; NVIDIA's driver stack handles kernel launches.
- For **MPS** (Apple Silicon), PyTorch uses its **Metal Performance Shaders backend**, which is less mature than MLX — not all ops are supported natively, with some falling back to CPU.
- Tensors are explicitly `.to(self.device)` — unlike MLX's UMA model, with CUDA there's a real host→device memory copy at load time.
- Model weights are loaded in `bfloat16`, unquantized, from a standard HuggingFace repo.
- **Stack:** `transformers` → `torch` → cuDNN/cuBLAS (CUDA) or MPS (Apple) → GPU

---

## Summary

| | **MLX** | **llama.cpp** | **PyTorch** |
|---|---|---|---|
| **Low-level GPU API** | Metal (via MLX) | Metal or CUDA (C++ kernels) | CUDA (cuDNN/cuBLAS) or MPS |
| **Language at GPU boundary** | Swift/C++ (MLX core) | C++ | C++/CUDA |
| **Memory model** | Unified (UMA, no copy) | Partial offload (`n_gpu_layers`) | Explicit device tensors |
| **Quantization** | Optional (`QuantizedLinear`) | Required (GGUF format) | None (bfloat16) |
| **Platform** | Apple Silicon only | Cross-platform | Cross-platform |
| **Python layer thickness** | Medium (mlx-lm API) | Thin (`ctypes` over C++) | Thick (PyTorch autograd engine) |

The key distinction: **llama.cpp** gives the most direct C++ → GPU path with explicit layer-level control; **MLX** leverages UMA to avoid memory copies entirely; **PyTorch/MPS** is the least optimized path on Apple Silicon since MPS support in PyTorch is still incomplete compared to MLX's native Metal integration.
