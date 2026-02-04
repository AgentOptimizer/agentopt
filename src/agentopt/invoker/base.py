from typing import Any, Dict, Protocol


class InvokerProtocol(Protocol):
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
