"""
Type definitions for agentopt.

All types are fully typed using Pydantic models.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union
from pydantic import BaseModel

if TYPE_CHECKING:
    from .model_proxy import ModelProxy


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


class EvaluationTask(BaseModel):
    """A single evaluation task with input and expected output."""

    messages: List[Message]
    expected_answer: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationTask":
        """Create from dictionary format."""
        messages = [
            Message(role=m.get("role", "user"), content=m.get("content", ""))
            for m in data.get("messages", [])
        ]
        return cls(
            messages=messages,
            expected_answer=data.get("expected_answer", ""),
        )

    def to_invoke_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for agent invocation."""
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages]
        }


# Type aliases
AccuracyFn = Callable[[str, str], bool]
ModelSpec = Union[str, Any]  # Model name string or model object
ModelsConfig = Dict["ModelProxy", List[ModelSpec]]
