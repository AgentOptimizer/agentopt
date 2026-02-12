"""Vanilla OpenAI Agents SDK summarization snippets (no AgentOpt)."""

from openai import OpenAI

from examples.sdk_shared import small_summary_dataset


def build_agent(model: str = "gpt-4o-mini"):
    client = OpenAI()
    return client.agents.create(
        name="Summarizer",
        model=model,
        instructions="Summarize the user message in one sentence.",
    )


def run_agent(agent, text: str) -> str:
    client = OpenAI()
    result = client.agents.runs.create_and_poll(agent_id=agent.id, input=text)
    if hasattr(result, "output") and result.output:
        return result.output[0].content[0].text
    return str(result)


def main() -> None:
    dataset = small_summary_dataset()
    agent = build_agent()

    for input_data, expected in dataset:
        summary = run_agent(agent, input_data["input"])
        print(f"Input: {input_data['input']}\nSummary: {summary}\nExpected contains: {expected}\n")


if __name__ == "__main__":
    main()
