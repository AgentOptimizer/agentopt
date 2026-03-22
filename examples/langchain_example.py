"""
Example: LangChain agent with agentopt.

Prerequisites:
    1. pip install langchain langchain-openai agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agentopt import ModelSelector


@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"Search results for: {query}"


TOOLS = [search]

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use tools when needed to answer questions concisely."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


class MyAgent:
    """LangChain tool-calling agent."""

    def __init__(self, models):
        llm = ChatOpenAI(model=models["agent"], disable_streaming=True)
        agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
        self.executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)

    def run(self, input_data):
        result = self.executor.invoke({"input": input_data})
        return result["output"]


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
]


if __name__ == "__main__":
    selector = ModelSelector(
        agent=MyAgent,
        models={
            "agent": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        },
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",
    )

    results = selector.select_best(parallel=True)
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
