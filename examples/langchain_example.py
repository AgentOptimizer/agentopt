import argparse
import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import matplotlib.pyplot as plt

from agentopt import ModelProxy, BruteForceModelSelector


def load_dataset(dataset_dir, filename):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for LangChain."""
    dataset_path = Path(dataset_dir)
    jsonl_file = dataset_path / filename

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Format input for LangChain's invoke({"input": ...})
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks


def eval_fn(expected, actual):
    return expected.lower() in str(actual.get("output", "")).lower()


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
    llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

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

    return llm, agent_executor


def multiagent_example():
    """
    Multi-agent setup with different LLMs for different agents.
    Uses a custom invoke_fn for sequential execution.
    """
    # 1. Wrap the LLMs
    researcher_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))
    coder_llm = ModelProxy(ChatOpenAI(model="gpt-4o-mini"))

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

    # 5. Register executors for automatic chain rebuild on model swap.
    # Framework is auto-detected; tools and prompt are extracted automatically.
    researcher_llm.register(researcher_executor)
    coder_llm.register(coder_executor)

    # 6. Custom invoke_fn that chains agents sequentially
    def chained_invoke(input_data):
        question = input_data["input"]
        research_result = researcher_executor.invoke({"input": question})
        research_output = research_result.get("output", "")
        coder_result = coder_executor.invoke(
            {"input": f"Based on this context: {research_output}\n\nTask: {question}"}
        )
        return coder_result

    return (researcher_llm, coder_llm), chained_invoke


def run_model_selection(
    agent_or_invoke_fn,
    llm_proxies,
    use_invoke_fn=False,
    parallel=False,
    dataset_file=None,
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)

    kwargs = {
        "models": {
            llm: [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "openai/gpt-5.1",
            ]
            for llm in llm_proxies
        },
        "eval_fn": eval_fn,
        "dataset": dataset,
    }

    if use_invoke_fn:
        kwargs["invoke_fn"] = agent_or_invoke_fn
    else:
        kwargs["agent"] = agent_or_invoke_fn

    selector = BruteForceModelSelector(**kwargs)

    results = selector.select_best(parallel=parallel)
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


EXAMPLES = {
    "single": ("Single-agent", single_agent_example),
    "multi": ("Multi-agent (chained)", multiagent_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain model selection example")
    parser.add_argument(
        "example",
        choices=EXAMPLES.keys(),
        help="Which example to run",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run model selection in parallel (single-agent only)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math_problems.jsonl",
        help="JSONL filename in examples/datasets/ (default: first .jsonl found)",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    mode = "parallel" if args.parallel else "sequential"

    print("=" * 40)
    print(f"{label} ({mode})")
    print("=" * 40)

    result = setup_fn()

    if args.example == "single":
        llm_proxy, agent_executor = result
        results = run_model_selection(
            agent_executor,
            [llm_proxy],
            parallel=args.parallel,
            dataset_file=args.dataset,
        )
    else:
        # Multi-agent returns (proxies_tuple, invoke_fn)
        llm_proxies, chained_invoke = result
        if args.parallel:
            print(
                "Warning: parallel mode not supported for multi-agent "
                "(uses invoke_fn). Running sequentially."
            )
        results = run_model_selection(
            chained_invoke,
            list(llm_proxies),
            use_invoke_fn=True,
            dataset_file=args.dataset,
        )

    plot_results(
        results, f"LangChain {label} Results", "examples/langchain_results.png"
    )
