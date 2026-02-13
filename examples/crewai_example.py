import json
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
)
import matplotlib.pyplot as plt

from agentopt import ModelProxy, BruteForceModelSelector


def load_dataset(dataset_dir):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for CrewAI."""
    dataset_path = Path(dataset_dir)
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in: {dataset_dir}")

    tasks = []
    with open(jsonl_files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Format input for CrewAI's kickoff(inputs={"input": ...})
            tasks.append(({"input": item["question"]}, item["output"]))
    return tasks


def eval_fn(expected, actual):
    return expected.lower() in str(actual).lower()


def without_opt_single_agent_example():
    llm = LLM(model="openai/gpt-4o-mini")

    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm,
        tools=[SerperDevTool(), WebsiteSearchTool()],
        verbose=False,
    )

    task = Task(
        description="{input}",
        expected_output="A clear answer",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    crew.kickoff(inputs={"input": "What is the capital of France?"})

    return crew


def single_agent_example():
    # 1. Wrap the LLM
    llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))

    # 2. Standard CrewAI setup - use proxy as llm
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm,
        tools=[SerperDevTool(), WebsiteSearchTool()],
        verbose=False,
    )

    task = Task(
        description="{input}",
        expected_output="A clear answer",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    return llm, crew


def multiagent_example():
    # 1. Wrap the LLM
    llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))

    # 2. Standard CrewAI setup - use proxy as llm
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm,
        tools=[SerperDevTool(), WebsiteSearchTool()],
        verbose=False,
    )

    sde = Agent(
        role="SDE",
        goal="Write efficient and correct code",
        backstory="You are a skilled software developer.",
        llm=llm,
        tools=[DirectoryReadTool(), FileReadTool()],
        verbose=False,
    )

    # Task 1: Research
    research_task = Task(
        description="{input}",
        expected_output="Research findings and analysis",
        agent=researcher,
    )

    # Task 2: Code (depends on research)
    code_task = Task(
        description="Based on the research findings, write code or provide a technical solution",
        expected_output="Working code or technical solution",
        agent=sde,
        context=[research_task],  # Uses output from research_task
    )

    crew = Crew(
        agents=[researcher, sde],
        tasks=[research_task, code_task],
        process=Process.sequential,
        verbose=False,
    )
    return llm, crew


def multiagent_multillm_example():
    # 1. Wrap the LLMs
    llm_research = ModelProxy(LLM(model="openai/gpt-4o-mini"))
    llm_sde = ModelProxy(LLM(model="openai/gpt-4o-mini"))

    # 2. Standard CrewAI setup - use proxies as llms
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm_research,
        verbose=False,
    )

    sde = Agent(
        role="SDE",
        goal="Write efficient and correct code",
        backstory="You are a skilled software developer.",
        llm=llm_sde,
        verbose=False,
    )

    # Task 1: Research
    research_task = Task(
        description="{input}",
        expected_output="Research findings and analysis",
        agent=researcher,
    )

    # Task 2: Code (depends on research)
    code_task = Task(
        description="Based on the research findings, write code or provide a technical solution",
        expected_output="Working code or technical solution",
        agent=sde,
        context=[research_task],  # Uses output from research_task
    )

    crew = Crew(
        agents=[researcher, sde],
        tasks=[research_task, code_task],
        process=Process.sequential,
        verbose=False,
    )
    return (llm_research, llm_sde), crew


def run_model_selection(crew, llm_proxies):
    dataset = load_dataset("examples/datasets")

    selector = BruteForceModelSelector(
        models={
            llm_proxy: [
                "openai/gpt-4o-mini",
                "openai/gpt-4.1-mini",
            ]
            for llm_proxy in llm_proxies
        },
        eval_fn=eval_fn,
        dataset=dataset,
        agent=crew,
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
    llm_proxy, crew = single_agent_example()
    results = run_model_selection(crew, [llm_proxy])

    # # Multi-agent example with single LLM
    # print("=" * 20)
    # print("Multi-agent example with single LLM")
    # print("=" * 20)
    # llm_proxy, crew = multiagent_example()
    # results = run_model_selection(crew, [llm_proxy])

    # print("=" * 20)
    # print("Multi-agent with multiple LLMs")
    # print("=" * 20)
    # llm_proxies, crew = multiagent_multillm_example()
    # results = run_model_selection(crew, llm_proxies)

    # Plot all results
    plot_results(
        results, "CrewAI Model Selection Results", "examples/crewai_results.png"
    )
