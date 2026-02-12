"""Vanilla OpenAI Agents SDK math QA loop (no AgentOpt)."""

from openai import OpenAI

from examples.sdk_shared import load_jsonl_dataset


def build_agent(model: str = "gpt-4o-mini"):
    client = OpenAI()
    return client.agents.create(
        name="Math QA Agent",
        model=model,
        instructions="Answer the user's math question concisely.",
    )


def run_agent(agent, question: str) -> str:
    client = OpenAI()
    result = client.agents.runs.create_and_poll(agent_id=agent.id, input=question)
    if hasattr(result, "output") and result.output:
        return result.output[0].content[0].text
    return str(result)


def main(dataset_dir: str = "examples/datasets") -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    agent = build_agent()

    for input_data, expected in dataset:
        answer = run_agent(agent, input_data["input"])
        print(f"Q: {input_data['input']}\nA: {answer}\nExpected: {expected}\n")


if __name__ == "__main__":
    main()
