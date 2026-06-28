from abc import ABC, abstractmethod
from pathlib import Path

from file_agent.document import Document


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        raise NotImplementedError
