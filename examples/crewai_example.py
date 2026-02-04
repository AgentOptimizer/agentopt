from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
)
import matplotlib.pyplot as plt

from agentopt import ModelProxy, ModelSelector, CrewInvoker


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

    # 3. Wrap crew with invoker
    invoker = CrewInvoker(crew)

    return llm, invoker


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


def hierarchical_example():
    """
    Hierarchical process: Manager agent dynamically delegates tasks to workers.
    Tasks don't need agent= assignment - the manager decides who does what.
    """
    # 1. Wrap both worker LLM and manager LLM
    worker_llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))
    manager_llm = ModelProxy(LLM(model="openai/gpt-4o"))

    # 2. Define worker agents
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=worker_llm,
        verbose=False,
    )

    sde = Agent(
        role="SDE",
        goal="Write efficient and correct code",
        backstory="You are a skilled software developer.",
        llm=worker_llm,
        verbose=False,
    )

    # 3. Define tasks WITHOUT agent= (manager will assign dynamically)
    research_task = Task(
        description="{input}",
        expected_output="Research findings and analysis",
    )

    code_task = Task(
        description="Based on the research findings, write code or provide a technical solution",
        expected_output="Working code or technical solution",
    )

    # 4. Create hierarchical crew with manager
    crew = Crew(
        agents=[researcher, sde],
        tasks=[research_task, code_task],
        process=Process.hierarchical,
        manager_llm=manager_llm,  # Manager decides who does what
        verbose=False,
    )

    invoker = CrewInvoker(crew)
    return (worker_llm, manager_llm), invoker


def accuracy_fn(expected: str, actual: str) -> bool:
    return expected.lower() in actual.lower()


def run_model_selection(invoker, llm_proxies):
    # 4. Model selection
    selector = ModelSelector(
        invoker=invoker,
        models={
            llm: [
                "openai/gpt-4o-mini",
                "openai/gpt-4.1-mini",
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

    # Multi-agent example with single LLM
    print("=" * 20)
    print("Multi-agent example with single LLM")
    print("=" * 20)
    llm_proxy, crew = multiagent_example()
    invoker = CrewInvoker(crew)
    results = run_model_selection(invoker, [llm_proxy])

    print("=" * 20)
    print("Multi-agent with multiple LLMs")
    print("=" * 20)
    llm_proxies, crew = multiagent_multillm_example()
    invoker = CrewInvoker(crew)
    results = run_model_selection(invoker, llm_proxies)

    print("=" * 20)
    print("Hierarchical example")
    print("=" * 20)
    llm_proxies, invoker = hierarchical_example()
    results = run_model_selection(invoker, llm_proxies)

    # Plot all results
    plot_results(
        results, "CrewAI Model Selection Results", "examples/crewai_results.png"
    )
