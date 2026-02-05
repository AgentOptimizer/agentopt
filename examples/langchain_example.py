import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root. Copy .env.example to .env and add your API keys.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from langchain_openai import ChatOpenAI


def _chat_openai(model: str, **kwargs):
    """ChatOpenAI using OpenRouter if OPENROUTER_API_KEY is set, else OpenAI."""
    if os.getenv("OPENROUTER_API_KEY"):
        return ChatOpenAI(
            model=model,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            **kwargs,
        )
    return ChatOpenAI(model=model, **kwargs)
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import matplotlib.pyplot as plt

from agentopt import (
    ModelProxy,
    ModelSelector,
    LangchainInvoker,
    ChainedLangchainInvoker,
)


# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2' or '10 * 5'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def single_agent_example():
    """Single agent with search and calculator tools."""
    # 1. Wrap the LLM
    llm = ModelProxy(_chat_openai("gpt-4o-mini"))

    # 2. Define tools
    search = DuckDuckGoSearchRun()
    tools = [search, calculator]

    # 3. Create agent prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful research assistant. Use tools when needed."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 4. Create agent and executor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 5. Wrap with invoker
    invoker = LangchainInvoker(agent_executor)

    return llm, invoker


def multiagent_example():
    """
    Multi-agent setup with different LLMs for different agents.
    Uses ChainedLangchainInvoker for sequential execution.
    """
    # 1. Wrap the LLMs
    researcher_llm = ModelProxy(_chat_openai("gpt-4o-mini"))
    coder_llm = ModelProxy(_chat_openai("gpt-4o-mini"))

    # 2. Define tools for each agent
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # 3. Create researcher agent
    researcher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant. Find information using search tools.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    researcher_agent = create_tool_calling_agent(
        researcher_llm, [search, wikipedia], researcher_prompt
    )
    researcher_executor = AgentExecutor(
        agent=researcher_agent, tools=[search, wikipedia], verbose=False
    )

    # 4. Create coder agent
    coder_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a coding assistant. Write efficient code based on research findings.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    coder_agent = create_tool_calling_agent(coder_llm, [calculator], coder_prompt)
    coder_executor = AgentExecutor(agent=coder_agent, tools=[calculator], verbose=False)

    # 5. Use ChainedLangchainInvoker for sequential execution
    invoker = ChainedLangchainInvoker([researcher_executor, coder_executor])

    return (researcher_llm, coder_llm), invoker


def accuracy_fn(expected: str, actual: str) -> bool:
    return expected.lower() in actual.lower()


def run_model_selection(invoker, llm_proxies):
    selector = ModelSelector(
        invoker=invoker,
        models={
            llm: [
                "gpt-4o-mini",
                "gpt-4o",
            ]
            for llm in llm_proxies
        },
        accuracy_fn=accuracy_fn,
        dataset_dir="examples/datasets",
    )

    results = selector.select_best()
    print(f"\nBest: {results.get_best()}")
    return results


def plot_results(results, title="Model Performance", save_path=None):
    """Plot accuracy vs latency for model selection results."""
    plt.figure(figsize=(10, 6))
    accuracies = [r.accuracy for r in results]
    latencies = [r.latency_seconds for r in results]
    names = [r.model_name for r in results]

    plt.scatter(latencies, accuracies)

    for name, lat, acc in zip(names, latencies, accuracies):
        plt.annotate(name, (lat, acc))

    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Single-agent example
    print("=" * 20)
    print("Single-agent example")
    print("=" * 20)
    llm_proxy, invoker = single_agent_example()
    results = run_model_selection(invoker, [llm_proxy])

    # Multi-agent example
    print("=" * 20)
    print("Multi-agent example")
    print("=" * 20)
    llm_proxies, invoker = multiagent_example()
    results = run_model_selection(invoker, llm_proxies)

    # Plot all results
    plot_results(
        results, "LangChain Model Selection Results", "examples/langchain_results.png"
    )
