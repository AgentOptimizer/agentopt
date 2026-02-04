"""
Type definitions for agentopt.

All types are fully typed using Pydantic models.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from pydantic import BaseModel, Field


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


class ModelResult(BaseModel):
    """Result of evaluating a single model."""

    model_name: str
    accuracy: float
    latency_seconds: float
    attribute: str
    is_best: bool = False


class SelectionResults(BaseModel):
    """Results from model selection."""

    results: List[ModelResult] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def get_best(self, attribute: Optional[str] = None) -> Optional[ModelResult]:
        """Get the best model result, optionally filtered by attribute."""
        filtered = self.results
        if attribute:
            filtered = [r for r in self.results if r.attribute == attribute]
        best = [r for r in filtered if r.is_best]
        return best[0] if best else None

    def get_by_attribute(self, attribute: str) -> List[ModelResult]:
        """Get all results for a specific attribute."""
        return [r for r in self.results if r.attribute == attribute]

    def to_csv(self, path: str) -> None:
        """Save results to CSV file."""
        import csv

        if not self.results:
            return

        fieldnames = [
            "model_name",
            "accuracy",
            "latency_seconds",
            "attribute",
            "is_best",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.model_dump())


class AgentProtocol(Protocol):
    """Protocol for agents that can be evaluated by ModelSelector."""

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent with input.

        Args:
            input_dict: Dictionary with "messages" key containing list of messages

        Returns:
            Dictionary with "messages" key containing response
        """
        ...


# Type aliases
AccuracyFn = Callable[[str, str], bool]
ModelSpec = Union[str, Any]  # Model name string or model object
ModelsConfig = Dict[str, List[ModelSpec]]
