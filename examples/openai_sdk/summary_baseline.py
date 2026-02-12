"""Vanilla OpenAI Agents SDK summarization snippets (no AgentOpt)."""

from openai import OpenAI

from examples.sdk_shared import small_summary_dataset


def main() -> None:
    dataset = small_summary_dataset()
    client = OpenAI()
    model = "gpt-4o-mini"

    for input_data, expected in dataset:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input_data["input"]}],
            max_tokens=128,
        )
        answer = response.choices[0].message.content
        print(f"Input: {input_data['input']}\nSummary: {answer}\nExpected contains: {expected}\n")


if __name__ == "__main__":
    main()
