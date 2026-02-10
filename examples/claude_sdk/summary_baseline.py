"""Vanilla Claude SDK summarization snippets (no AgentOpt)."""

from anthropic import Anthropic

from examples.sdk_shared import small_summary_dataset


def main() -> None:
    dataset = small_summary_dataset()
    client = Anthropic()
    model = "claude-3-5-haiku-latest"

    for input_data, expected in dataset:
        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": input_data["input"]}],
        )
        answer = message.content[0].text if message.content else ""
        print(f"Input: {input_data['input']}\nSummary: {answer}\nExpected contains: {expected}\n")


if __name__ == "__main__":
    main()
