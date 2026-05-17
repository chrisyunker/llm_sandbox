import argparse
import yaml
import logging
from src.logger import set_log_level, cprint

class Config:
    def __init__(self, path: str):
        self.path = path

    def load_and_parse_args(self, defaults: dict):
        self.load()
        self.parse_args(defaults)
        self.merge_impl_section()

    def parse_args(self, defaults: dict):
        parser = argparse.ArgumentParser(description='Run local LLM models')
        parser.add_argument('-pr', '--prompt',
                            type=str,
                            required=True,
                            help='Prompt string')
        parser.add_argument('-mo', '--model',
                            type=str,
                            help='Model nickname (defined in config)')
        parser.add_argument('-mt', '--max_tokens',
                            type=int,
                            help='Max tokens')
        parser.add_argument('-temp', '--temperature',
                            type=float,
                            help='Model temperature')
        parser.add_argument('--top-k',
                            type=int,
                            help='Model Top-k')
        parser.add_argument('--top-p',
                            type=float,
                            help='Model Top-p')
        parser.add_argument('--min-p',
                            type=float,
                            help='Model Min-p')
        parser.add_argument('--gpu-layers',
                            type=int,
                            help='Model GPU Layers (llama)')
        parser.add_argument('-im', '--impl',
                            choices=['mlx', 'llama', 'torch'],
                            required=True,
                            help='Implementation library')
        parser.add_argument('-l', '--log',
                            choices=['error', 'warning', 'info', 'debug'],
                            type=str,
                            default="info",
                            help='Log Level')
        parser.add_argument('--cpu',
                            action='store_true',
                            help='Run in CPU (torch)')
        parser.add_argument('--inc',
                            action='store_true',
                            help='Incremental generation')
        parser.suggest_on_error = True
        args = parser.parse_args()

        set_log_level(args.log)

        for k, v in vars(args).items():
            if v is not None:
                logging.debug(f"Setting cfg key: {k} to value: {v}")
                setattr(self, k, v)

        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)


    def merge_impl_section(self):
        impl = getattr(self, "impl", None)
        if not impl or not hasattr(self, impl):
            return
        impl_cfg = getattr(self, impl)
        self.__dict__.update({k: v for k, v in impl_cfg.items() if k != "models"})
        self._merge_model_nickname(impl_cfg)

    def _merge_model_nickname(self, impl_cfg: dict):
        models = impl_cfg.get("models")
        if not models:
            return
        nickname = getattr(self, "model", None)
        if nickname is None:
            if len(models) == 1:
                nickname = next(iter(models))
            else:
                names = ", ".join(models.keys())
                raise SystemExit(f"Multiple models available; select one with --model: {names}")
        if nickname not in models:
            names = ", ".join(models.keys())
            raise SystemExit(f"Unknown model nickname '{nickname}'. Available: {names}")
        self.__dict__.update(models[nickname])

    def load(self):
        with open(self.path, "r") as f:
            self.__dict__.update(yaml.safe_load(f))

    def print(self):
        cprint("bbl",
               "===================\n"
               "Configuration:\n"
               f"Max tokens:    {self.max_tokens}\n"
               f"Temperature:   {self.temperature}\n"
               f"Top-k:         {self.top_k}\n"
               f"Top-p:         {self.top_p}\n"
               f"Min-p:         {self.min_p}\n"
               f"CTX Window:    {self.ctx_window}\n"
               f"GPU Layers:    {self.gpu_layers}\n"
               f"Run in CPU:    {self.cpu}\n"
               "===================")