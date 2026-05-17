import logging
import warnings
from src.mlx_harness import MLXHarness
from src.llama_harness import LlamaHarness
from src.torch_harness import TorchHarness
from src.config import Config
from src.logger import setup_logging, cprint

CONFIG_FILE = "config.yaml"
DEFAULTS = {
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.7,
    "min_p": 0.05,
    "gpu_layers": 2,
}

def main():
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", module="huggingface_hub.utils._validators")

    cfg = Config(CONFIG_FILE)
    cfg.load_and_parse_args(DEFAULTS)

    if cfg.impl == 'mlx':
        harness = MLXHarness()
    elif cfg.impl == 'llama':
        harness = LlamaHarness()
    elif cfg.impl == 'torch':
        harness = TorchHarness()
    else:
        raise SystemExit(f"Unknown implementation: {cfg.impl}")

    logging.info(f"Loading model: {cfg.model_name}")
    harness.load(cfg)
    logging.info(f"Model loaded in {cfg.load_time:.2f} seconds")

    cfg.print()
    harness.print_model_stats()

    logging.info("Generating response....")
    cprint("byl",
           "===================\n"
           "Prompt:\n"
           f"{cfg.prompt.strip()}\n"
           "===================")

    response = harness.generate(cfg)
    logging.info(f"Response generated in {cfg.infer_time:.2f} seconds")

    cprint("bcn",
           "===================\n"
           "Response:\n"
           f"{response}\n"
           "===================")

if __name__ == "__main__":
    main()
