"""What an agent run produced: the final answer plus everything it did on the way."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class Trajectory:
    final: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    error: str | None = None

    def cost(self, usd_per_mtok_in: float = 3.0, usd_per_mtok_out: float = 15.0) -> float:
        return (self.tokens_in * usd_per_mtok_in + self.tokens_out * usd_per_mtok_out) / 1e6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d = dict(d)
        d["tool_calls"] = [ToolCall(**c) for c in d.get("tool_calls", [])]
        return cls(**d)
