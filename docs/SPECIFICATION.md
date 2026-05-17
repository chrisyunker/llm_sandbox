# LLM Sandbox Specification

## Overview

LLM Sandbox is a CLI tool for running local LLM inference experiments across three backend implementations: Apple MLX, llama.cpp, and PyTorch. It provides a consistent interface for loading models, running generation, and comparing behavior across backends.

## CLI Usage

```
python main.py -pr <prompt> -im <impl> [options]
```

### Required Arguments

| Flag | Description |
|------|-------------|
| `-pr` / `--prompt` | Prompt string to send to the model |
| `-im` / `--impl` | Backend implementation: `mlx`, `llama`, or `torch` |

### Optional Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-mo` / `--model` | Model nickname defined in `config.yaml`. If omitted and only one model exists for the impl, it is selected automatically. | — |
| `-mt` / `--max_tokens` | Maximum tokens to generate | 1000 |
| `-tp` / `--temperature` | Sampling temperature | 0.7 |
| `-l` / `--log` | Log level: `error`, `warning`, `info`, `debug` | `info` |
| `--inc` | Incremental (step-by-step) generation mode; prints each token as it is produced | off |

## Architecture

The codebase uses an abstract `Harness` pattern. `main.py` instantiates the appropriate concrete harness based on `--impl`, then calls a fixed three-step sequence:

```
harness.load(cfg)         # load model weights
harness.print_model_stats()  # display parameter counts, memory, etc.
harness.generate(cfg)     # run inference and return response string
```

Configuration is loaded from `config.yaml`, merged with CLI arguments, and passed as a `Config` object to each harness method. Timing for both load and inference is written back into `cfg.load_time` and `cfg.infer_time` by the harness.

## Components

### `main.py`

Entry point. Initializes logging, loads and parses config, selects the harness, and drives the load → stats → generate sequence. Prints the prompt and response to stdout with color formatting.

### `src/config.py` — `Config`

Handles all configuration. `load_and_parse_args()` performs three steps in order:

1. **`load()`** — reads `config.yaml` into `self.__dict__` via `yaml.safe_load`.
2. **`parse_args()`** — parses CLI arguments with `argparse`; sets the log level immediately; merges non-`None` args and any missing defaults into `self`.
3. **`merge_impl_section()`** — merges the impl-specific block from the YAML (e.g., the `mlx:` section) into `self`, then resolves the model nickname via `_merge_model_nickname()`, which merges the selected model's fields (e.g., `model_name`, `model_type`) into `self`.

`Config.print()` displays the active sampling parameters in color.

### `src/harness.py` — `Harness`

Abstract base class (ABC) defining the three-method interface all backends must implement:

- `load(cfg: Config) -> None`
- `generate(cfg: Config) -> str`
- `print_model_stats() -> None`

### `src/mlx_harness.py` — `MLXHarness`

Backend using Apple's [mlx-lm](https://github.com/ml-explore/mlx-lm) library, intended for Apple Silicon Macs.

- **`load`**: calls `mlx_lm.load(cfg.model_name)` to get model and tokenizer.
- **`print_model_stats`**: reports total parameter count, vocab size, whether any layers are quantized (`QuantizedLinear`), and active/peak Metal GPU memory.
- **`generate`**: applies the chat template, then either:
  - **Normal mode**: calls `mlx_lm.generate` with a sampler built from `temperature`, `top_k`, `top_p`, `min_p`.
  - **Incremental mode** (`--inc`): steps through `generate_step` token by token, printing each step's sequence and next token, stopping at EOS.

### `src/llama_harness.py` — `LlamaHarness`

Backend using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), which runs GGUF-quantized models.

- **`load`**: calls `Llama.from_pretrained` with `repo_id`, `filename` glob (`model_type`), context window (`ctx_window`), GPU layers (`gpu_layers`), and sampling params. Verbose llama.cpp output is enabled only at `debug` log level.
- **`print_model_stats`**: reads model metadata to display name, vocab size, context size, and embedding dimension.
- **`generate`**: calls `create_chat_completion` with the prompt as a user message; extracts the response string from the choices array.

### `src/torch_harness.py` — `TorchHarness`

Backend using [Hugging Face Transformers](https://huggingface.co/docs/transformers) with PyTorch.

- **`load`**: detects the best available device via `util.get_device()`, then loads tokenizer and model (`bfloat16`, auto device map).
- **`print_model_stats`**: reports parameter count, dtype, device, vocab size, layer count, hidden size, and allocated GPU/MPS memory.
- **`generate`**: applies the chat template, then either:
  - **Normal mode**: calls `model.generate` with sampling parameters.
  - **Incremental mode** (`--inc`): manually steps through the model using KV-cache (`past_key_values`), calling the local `_sample_token` function at each step. Prints each step's sequence and next token, stopping at EOS.

`_sample_token(logits, temperature, top_k, top_p, min_p)` is a module-level function that applies temperature scaling, top-k filtering, min-p filtering, and top-p (nucleus) filtering in sequence, then samples from the resulting distribution.

### `src/logger.py`

Provides colored terminal output.

- **`Color`** (Enum): ANSI escape codes for 12 colors (normal + bright variants).
- **`ColorLogger`**: a `logging.Formatter` subclass that colorizes `WARNING` messages yellow and `ERROR`/`CRITICAL` messages red.
- **`setup_logging(level)`**: configures the root logger with a `ColorLogger` handler.
- **`set_log_level(level)`**: updates the root logger's level at runtime (called after CLI args are parsed).
- **`cprint(color, message, ...)`**: prints a message in the given color using a short two-or-three-character code (e.g., `"byl"` = bright yellow, `"bcn"` = bright cyan).

### `src/util.py`

Utility functions for the PyTorch backend.

- **`get_device()`**: returns `"mps"`, `"cuda"`, or `"cpu"` based on hardware availability.
- **`print_memory_usage()`**: logs system RAM usage and whether MPS is active.
- **`signal_handler` / `setup_signal_handlers()`**: registers a `SIGINT` handler for graceful Ctrl-C exit.

## Configuration File (`config.yaml`)

The YAML file has one top-level section per implementation (`mlx`, `llama`, `torch`), plus global sampling defaults.

```yaml
mlx:
  models:
    <nickname>:
      model_name: "<HuggingFace repo>"
llama:
  models:
    <nickname>:
      model_name: "<HuggingFace GGUF repo>"
      model_type: "<filename glob, e.g. *Q8_0.gguf>"
torch:
  models:
    <nickname>:
      model_name: "<HuggingFace repo>"

# Global sampling defaults (overridden by CLI args)
max_tokens: 1000
temperature: 0.7
top_k: 40
min_p: 0.05
ctx_window: 1024   # llama only: context window size
gpu_layers: 2      # llama only: layers offloaded to GPU
```

Model nicknames are passed via `--model`. If an impl section has exactly one model, it is selected automatically when `--model` is omitted.
