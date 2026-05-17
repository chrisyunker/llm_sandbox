import time
from llama_cpp import Llama
from .harness import Harness
from .config import Config
from .logger import cprint

class LlamaHarness(Harness):
    def __init__(self):
        self.model = None

    def load(self, cfg: Config):
        start = time.perf_counter()

        self.model = Llama.from_pretrained(
            repo_id=cfg.model_name,
            filename=cfg.model_type,
            n_ctx=cfg.ctx_window,
            n_gpu_layers=cfg.gpu_layers,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            min_p=cfg.min_p,
            seed=42,
            verbose=(cfg.log == "debug")
        )

        end = time.perf_counter()
        cfg.load_time = end - start

    def print_model_stats(self):
        meta = self.model.metadata
        cprint("bbl",
               "===================\n"
               "Model:\n"
               f"Name:          {meta.get('general.name', 'unknown')}\n"
               f"Vocab size:    {self.model.n_vocab()}\n"
               f"Context size:  {self.model.n_ctx()}\n"
               f"Embedding dim: {self.model.n_embd()}\n"
               "===================")

    def generate(self, cfg: Config):
        start = time.perf_counter()

        output = self.model.create_chat_completion(
            messages=[{"role": "user", "content": cfg.prompt}],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature
        )
        response = output["choices"][0]["message"]["content"]

        end = time.perf_counter()
        cfg.infer_time = end - start
        return response
    