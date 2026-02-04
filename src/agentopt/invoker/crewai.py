"""
CrewAI wrapper for model selection compatibility.

Provides CrewInvoker, a minimal wrapper that makes a Crew invokable
in a way that is compatible with ModelSelector.
"""

from typing import Any, Dict

from .base import InvokerProtocol


class CrewInvoker(InvokerProtocol):
    """
    Minimal wrapper that makes a Crew invokable for ModelSelector.

    Usage:
        llm = ModelProxy(LLM(model='gpt-4o'))
        agent = Agent(role="...", llm=llm, ...)
        task = Task(description="{input}", expected_output="...", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])

        invoker = CrewInvoker(crew)

        selector = ModelSelector(
            invoker=invoker,
            models={llm: ["openai/gpt-4o-mini", "openai/gpt-4o"]},
            accuracy_fn=check_accuracy,
        )
    """

    def __init__(self, crew: Any) -> None:
        self.crew = crew

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the crew with input messages."""
        messages = input_dict.get("messages", [])
        question = messages[0].get("content", "") if messages else ""

        # Pass input to crew via kickoff inputs
        result = self.crew.kickoff(inputs={"input": question})
        return {"messages": [{"role": "assistant", "content": str(result)}]}
