from abc import ABC, abstractmethod
from .config import Config

class Harness(ABC):

    @abstractmethod
    def load(self, cfg: Config) -> None: ...
    
    @abstractmethod
    def generate(self, cfg: Config) -> str: ...

    @abstractmethod
    def print_model_stats(self) -> None: ...
