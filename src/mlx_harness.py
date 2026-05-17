import time
import mlx_lm
import logging
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
from .harness import Harness
from .config import Config
from .logger import cprint

class MLXHarness(Harness):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self, cfg: Config):
        start = time.perf_counter()

        self.model, self.tokenizer = mlx_lm.load(cfg.model_name)
        logging.debug(f"Model: {self.model}")

        end = time.perf_counter()
        cfg.load_time = end - start

    def print_model_stats(self):
        from mlx.utils import tree_flatten
        import mlx.nn as nn
        total = sum(v.size for _, v in tree_flatten(self.model.parameters()))
        active = mx.get_active_memory() / 1e9
        peak = mx.get_peak_memory() / 1e9
        quantized = any(isinstance(m, nn.QuantizedLinear) for _, m in tree_flatten(self.model))
        cprint("bbl",
               "===================\n"
               "Model:\n"
               f"Parameters:    {total / 1e9:.2f}B\n"
               f"Vocab size:    {self.tokenizer.vocab_size}\n"
               f"Quantized:     {quantized}\n"
               f"Metal memory:  active={active:.2f}GB  peak={peak:.2f}GB\n"
               "===================")

    def generate(self, cfg: Config):
        messages = [{"role": "user", "content": cfg.prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if cfg.log == "debug":
            cprint("byl",
                   "===================\n"
                   "Formatted Prompt:\n"
                   f"{formatted_prompt.rstrip()}\n"
                   "===================")

        start = time.perf_counter()

        if cfg.inc:
            input_ids = self.tokenizer.encode(formatted_prompt)
            generated_ids = list(input_ids)
            sampler = make_sampler(
                temp=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                min_p=cfg.min_p,
            )

            for step, (token, _logprobs) in enumerate(generate_step(
                mx.array(input_ids), self.model, max_tokens=cfg.max_tokens, sampler=sampler
            )):
                token_id = token if isinstance(token, int) else token.item()
                token_text = self.tokenizer.decode([token_id])

                cprint("byl",
                       "--------------------------------\n"
                       f"Generation Step: {step + 1}\n"
                       f"Current sequence length: {len(generated_ids)}\n"
                       f"Current sequence: [{self.tokenizer.decode(generated_ids)}]\n"
                       f"Next token: [{token_text}]\n"
                       "--------------------------------")

                generated_ids.append(token_id)
                if token_id == self.tokenizer.eos_token_id:
                    cprint("byl", "Hit EOS token, stopping generation")
                    break

            response = self.tokenizer.decode(generated_ids[len(input_ids):])
        else:
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=cfg.max_tokens,
                sampler=make_sampler(
                    temp=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    min_p=cfg.min_p,
                ),
                verbose=False
            )

        end = time.perf_counter()
        cfg.infer_time = end - start
        return response
