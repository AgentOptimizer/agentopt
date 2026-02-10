"""Vanilla Claude SDK math QA loop (no AgentOpt)."""

from anthropic import Anthropic

from examples.sdk_shared import load_jsonl_dataset


def main(dataset_dir: str = "examples/datasets") -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    client = Anthropic()
    model = "claude-3-5-haiku-latest"

    for input_data, expected in dataset:
        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": input_data["input"]}],
        )
        answer = message.content[0].text if message.content else ""
        print(f"Q: {input_data['input']}\nA: {answer}\nExpected: {expected}\n")


if __name__ == "__main__":
    main()
