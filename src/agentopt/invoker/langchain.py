"""
LangChain wrappers for model selection compatibility.

Provides LangchainInvoker (minimal wrapper) and ChainedLangchainInvoker
(multi-agent sequential) compatible with ModelSelector.
"""

from typing import Any, Dict, List

from .base import InvokerProtocol


class LangchainInvoker(InvokerProtocol):
    """
    Minimal wrapper that makes a LangChain AgentExecutor invokable for ModelSelector.

    Usage:
        from langchain_openai import ChatOpenAI
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

        llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)

        invoker = LangchainInvoker(agent_executor)

        selector = ModelSelector(
            invoker=invoker,
            models={llm: ["gpt-4o-mini", "gpt-4o"]},
            accuracy_fn=check_accuracy,
        )
    """

    def __init__(self, agent_executor: Any) -> None:
        self.agent_executor = agent_executor

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with input messages."""
        messages = input_dict.get("messages", [])
        question = messages[0].get("content", "") if messages else ""

        result = self.agent_executor.invoke({"input": question})
        return {
            "messages": [
                {"role": "assistant", "content": str(result.get("output", ""))}
            ]
        }


class ChainedLangchainInvoker(LangchainInvoker):
    """
    Chains multiple LangChain AgentExecutors sequentially.

    Each agent's output is fed as context to the next agent.

    Usage:
        researcher_executor = AgentExecutor(agent=researcher_agent, tools=[search])
        coder_executor = AgentExecutor(agent=coder_agent, tools=[calculator])

        invoker = ChainedLangchainInvoker([researcher_executor, coder_executor])

        selector = ModelSelector(
            invoker=invoker,
            models={researcher_llm: [...], coder_llm: [...]},
            accuracy_fn=check_accuracy,
        )
    """

    def __init__(self, agent_executors: List[Any]) -> None:
        self.agent_executors = agent_executors
        # Set first executor as the main one for compatibility
        super().__init__(agent_executors[0] if agent_executors else None)

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke agents sequentially, passing output as context to next agent."""
        messages = input_dict.get("messages", [])
        question = messages[0].get("content", "") if messages else ""

        current_input = question
        result_output = ""

        for i, executor in enumerate(self.agent_executors):
            if i == 0:
                # First agent gets the original question
                result = executor.invoke({"input": current_input})
            else:
                # Subsequent agents get previous output as context
                chained_input = (
                    f"Based on this context: {result_output}\n\nTask: {question}"
                )
                result = executor.invoke({"input": chained_input})

            result_output = result.get("output", "")

        return {"messages": [{"role": "assistant", "content": str(result_output)}]}
