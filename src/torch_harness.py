import os
import time
import torch
import torch.nn.functional as F
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from .harness import Harness
from .config import Config
from .util import get_device
from .logger import cprint

def _sample_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, min_p: float) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits).unsqueeze(0)
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < cutoff, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    if min_p > 0:
        threshold = min_p * probs.max()
        filtered = probs.masked_fill(probs < threshold, 0)
        if filtered.sum() > 0:
            probs = filtered / filtered.sum()
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cum - sorted_probs > top_p] = 0
        total = sorted_probs.sum()
        if total > 0:
            sorted_probs /= total
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    return torch.multinomial(probs, num_samples=1)

class TorchHarness(Harness):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self, cfg: Config):
        self.device = get_device()
        logging.debug(f"Using device: {self.device}")

        start = time.perf_counter()

        if cfg.model_name.endswith(".gguf"):
            model_base = getattr(cfg, "model_base", None)
            if model_base:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_base,
                    clean_up_tokenization_spaces=False,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_base,
                    torch_dtype="auto",
                    device_map=self.device,
                )
            else:
                model_dir = os.path.dirname(cfg.model_name)
                gguf_file = os.path.basename(cfg.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    gguf_file=gguf_file,
                    clean_up_tokenization_spaces=False,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    gguf_file=gguf_file,
                    device_map=self.device,
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name,
                clean_up_tokenization_spaces=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                dtype="auto",
                device_map=self.device,
            )

        end = time.perf_counter()
        cfg.load_time = end - start

    def print_model_stats(self):
        total = sum(p.numel() for p in self.model.parameters())
        dtype = next(self.model.parameters()).dtype
        cfg = self.model.config
        cprint("bbl",
               "===================\n"
               "Model:\n"
               f"Parameters:    {total / 1e9:.2f}B\n"
               f"Dtype:         {dtype}\n"
               f"Device:        {self.device}\n"
               f"Vocab size:    {self.tokenizer.vocab_size}\n"
               f"Layers:        {cfg.num_hidden_layers}\n"
               f"Hidden size:   {cfg.hidden_size}\n")
        if self.device == "mps":
            active = torch.mps.current_allocated_memory() / 1e9
            cprint("bbl", f"MPS memory:    active={active:.2f}GB")
        elif self.device == "cuda":
            active = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            cprint("bbl", f"CUDA memory:   active={active:.2f}GB  peak={peak:.2f}GB")
        cprint("bbl", "===================")

    def generate(self, cfg: Config):

        messages = [{"role": "user", "content": cfg.prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )

        if cfg.log == "debug":
            cprint("byl",
                   "===================\n"
                   "Formatted Prompt:\n"
                   f"{formatted_prompt.rstrip()}\n"
                   "===================")

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        start = time.perf_counter()

        if cfg.inc:
            input_ids = inputs.input_ids.clone()
            past_key_values = None
            for step in range(cfg.max_tokens):
                cprint("byl",
                       "--------------------------------\n"
                       f"Generation Step: {step + 1}\n"
                       f"Current sequence length: {input_ids.shape[1]}\n"
                       f"Current sequence: [{self.tokenizer.decode(input_ids[0])}]")

                with torch.no_grad():
                    model_input = input_ids if past_key_values is None else input_ids[:, -1:]
                    outputs = self.model(model_input, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                next_token_logits = logits[0, -1, :]
                next_token_id = _sample_token(
                    next_token_logits, cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p
                ).unsqueeze(0)
                next_token_text = self.tokenizer.decode(next_token_id[0])

                cprint("byl", f"Next token: [{next_token_text}]")
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                cprint("byl", "--------------------------------")

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    cprint("byl", "Hit EOS token, stopping generation")
                    break

            prompt_len = inputs.input_ids.shape[1]
            response = self.tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)

        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    min_p=cfg.min_p,
                    do_sample=True,
                    eos_token_id=self.model.config.eos_token_id
                )
            
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        end = time.perf_counter()
        cfg.infer_time = end - start
        return response
