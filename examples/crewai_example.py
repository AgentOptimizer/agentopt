import argparse
import json
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import (  # noqa: F401 — available for tool-based examples
    SerperDevTool,
    WebsiteSearchTool,
)
import matplotlib.pyplot as plt

from agentopt import ModelProxy, BruteForceModelSelector
from agentopt.model_selection import (
    ArmEliminationModelSelector,
    HillClimbingModelSelector,
)


SELECTORS = {
    "brute_force": BruteForceModelSelector,
    "arm_elimination": ArmEliminationModelSelector,
    "hill_climbing": HillClimbingModelSelector,
}


def load_dataset(dataset_dir, filename):
    """Load JSONL dataset and return (input_data, expected_answer) tuples for CrewAI."""
    dataset_path = Path(dataset_dir)
    jsonl_file = dataset_path / filename

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
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
    print("  [setup] proxy created (initial model: openai/gpt-4o-mini)")

    # 2. Standard CrewAI setup — no tools so the agent answers directly
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    print("  [setup] agent created: Researcher")

    task = Task(
        description="{input}",
        expected_output="A clear, concise answer",
        agent=researcher,
    )
    print("  [setup] task created: research task")

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    print("  [setup] crew assembled: 1 agent, 1 task")

    return llm, crew


def multiagent_example():
    # 1. Wrap the LLM
    llm = ModelProxy(LLM(model="openai/gpt-4o-mini"))
    print(
        "  [setup] proxy created (initial model: openai/gpt-4o-mini, shared by all agents)"
    )

    # 2. Standard CrewAI setup — no tools so agents answer directly
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    print("  [setup] agent created: Researcher")

    sde = Agent(
        role="SDE",
        goal="Write efficient and correct code",
        backstory="You are a skilled software developer.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    print("  [setup] agent created: SDE")

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
    print("  [setup] crew assembled: 2 agents, 2 tasks (research -> code)")
    return llm, crew


def multiagent_multillm_example():
    # 1. Wrap the LLMs
    llm_research = ModelProxy(LLM(model="openai/gpt-4o-mini"))
    print("  [setup] llm_research proxy created (initial model: openai/gpt-4o-mini)")
    llm_sde = ModelProxy(LLM(model="openai/gpt-4o-mini"))
    print("  [setup] llm_sde proxy created (initial model: openai/gpt-4o-mini)")

    # 2. Standard CrewAI setup — no tools so agents answer directly
    researcher = Agent(
        role="Researcher",
        goal="Find accurate information on any topic",
        backstory="You are an expert researcher with years of experience.",
        llm=llm_research,
        verbose=False,
        max_iter=5,
    )
    print("  [setup] agent created: Researcher (using llm_research proxy)")

    sde = Agent(
        role="SDE",
        goal="Write efficient and correct code",
        backstory="You are a skilled software developer.",
        llm=llm_sde,
        verbose=False,
        max_iter=5,
    )
    print("  [setup] agent created: SDE (using llm_sde proxy)")

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
    print("  [setup] crew assembled: 2 agents, 2 tasks, independent proxies")
    return (llm_research, llm_sde), crew


def run_model_selection(
    crew,
    llm_proxies,
    parallel: bool = False,
    dataset_file=None,
    selector_name: str = "brute_force",
):
    dataset = load_dataset("examples/datasets", filename=dataset_file)
    print(f"  [run] dataset loaded: {len(dataset)} samples from {dataset_file}")
    model_candidates = ["openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-5.1"]
    mode = "parallel" if parallel else "sequential"
    print(f"  [run] starting model selection ({mode}) — candidates: {model_candidates}")

    SelectorCls = SELECTORS[selector_name]
    selector = SelectorCls(
        models={
            llm_proxy: [
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "openai/gpt-5.1",
            ]
            for llm_proxy in llm_proxies
        },
        eval_fn=eval_fn,
        dataset=dataset,
        agent=crew,
    )

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
    plt.close()


EXAMPLES = {
    "single": ("Single-agent", single_agent_example),
    "multi": ("Multi-agent (shared LLM)", multiagent_example),
    "multi-llm": ("Multi-agent (separate LLMs)", multiagent_multillm_example),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrewAI model selection example")
    parser.add_argument(
        "example",
        choices=EXAMPLES.keys(),
        help="Which example to run",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run model selection in parallel",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="research_code_problems.jsonl",
        help="JSONL filename in examples/datasets/ (default: first .jsonl found)",
    )
    parser.add_argument(
        "--selector",
        choices=sorted(SELECTORS.keys()),
        default="brute_force",
        help="Model selector to use (default: brute_force)",
    )
    args = parser.parse_args()

    label, setup_fn = EXAMPLES[args.example]
    mode = "parallel" if args.parallel else "sequential"

    print("=" * 40)
    print(f"{label} ({mode})")
    print("=" * 40)

    print("\n[1] Setting up agents...")
    result = setup_fn()
    # multi-llm returns a tuple of proxies; the others return a single proxy
    if isinstance(result[0], tuple):
        llm_proxies, crew = result
    else:
        llm_proxy, crew = result
        llm_proxies = [llm_proxy]

    print("\n[2] Running model selection...")
    results = run_model_selection(
        crew,
        llm_proxies,
        parallel=args.parallel,
        dataset_file=args.dataset,
        selector_name=args.selector,
    )

    print("\n[3] Saving results plot...")
    plot_results(results, f"CrewAI {label} Results", "examples/crewai_results.png")
