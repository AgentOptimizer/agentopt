"""Vanilla Claude Agent SDK math QA loop (no AgentOpt)."""

from claude_agent_sdk import ClaudeAgentOptions

from .utils import load_jsonl_dataset, run_query_sync


def main(dataset_dir: str = "examples/datasets") -> None:
    dataset = load_jsonl_dataset(dataset_dir)
    agent_cfg = ClaudeAgentOptions(model="claude-3-5-haiku-latest")

    for input_data, expected in dataset:
        answer = run_query_sync(input_data["input"], agent_cfg.model)
        print(f"Q: {input_data['input']}\nA: {answer}\nExpected: {expected}\n")


if __name__ == "__main__":
    main()
