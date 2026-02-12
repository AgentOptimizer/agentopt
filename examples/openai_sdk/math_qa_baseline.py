"""Vanilla OpenAI Agents SDK math QA loop (no AgentOpt)."""

from agents import Agent, Runner

from .utils import load_jsonl_dataset


def build_agent(model: str = "gpt-4o-mini") -> Agent:
    return Agent(
        name="Math QA Agent",
        model=model,
        instructions="Answer the user's math question concisely.",
    )


def run_agent(agent: Agent, question: str) -> str:
    result = Runner.run_sync(agent, question)
    return result.final_output if hasattr(result, "final_output") else str(result)


def main(dataset_dir: str = "examples/datasets") -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    agent = build_agent()

    for input_data, expected in dataset:
        answer = run_agent(agent, input_data["input"])
        print(f"Q: {input_data['input']}\nA: {answer}\nExpected: {expected}\n")


if __name__ == "__main__":
    main()
