from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    id: str
    text: str
    type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    file_name: str
    file_type: str
    blocks: list[Block]
