"""Vanilla OpenAI Agents SDK math QA loop (no AgentOpt)."""

from openai import OpenAI

from examples.sdk_shared import load_jsonl_dataset


def main(dataset_dir: str = "examples/datasets") -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    client = OpenAI()
    model = "gpt-4o-mini"

    for input_data, expected in dataset:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input_data["input"]}],
            max_tokens=128,
        )
        answer = response.choices[0].message.content
        print(f"Q: {input_data['input']}\nA: {answer}\nExpected: {expected}\n")


if __name__ == "__main__":
    main()
