"""Demo: EvalCache with real LLM calls via OpenRouter.

Run twice to see caching in action:
  uv run python examples/cache_demo.py          # first run: all misses
  uv run python examples/cache_demo.py          # second run: all hits
  rm .cache/demo_eval.json                      # clear cache
"""

import json
import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agentopt import EvalCache, ModelProxy, ModelSelector

load_dotenv()

# OpenRouter is OpenAI-compatible; set OPENAI_API_KEY so model validation passes.
if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

CACHE_PATH = ".cache/demo_eval.json"

# Track how many LLM calls are actually made.
call_count = 0


def _make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )


def main():
    global call_count

    import json as _json

    with open("examples/datasets/research_code_problems.jsonl") as f:
        dataset = [
            (item["question"], item["output"])
            for line in f
            if (item := _json.loads(line.strip())) and line.strip()
        ]

    proxy = ModelProxy(_make_llm("openai/gpt-4o-mini"))
    # Pass pre-built model objects so set_model preserves base_url.
    candidates = [_make_llm("openai/gpt-4o-mini"), _make_llm("openai/gpt-4o")]

    cache = EvalCache(CACHE_PATH)
    print(f"Cache: {CACHE_PATH} ({len(cache)} existing entries)")

    def invoke_fn(input_data):
        """Call the LLM through the proxy."""
        global call_count
        call_count += 1
        question = input_data if isinstance(input_data, str) else input_data[0]
        response = proxy.invoke(f"Answer with just the number: {question}")
        return response.content.strip()

    def eval_fn(expected, actual):
        return 1.0 if expected in str(actual) else 0.0

    print(f"\nRunning model selection with {len(candidates)} candidates...")
    call_count = 0
    start = time.time()

    selector = ModelSelector(
        models={proxy: candidates},
        eval_fn=eval_fn,
        dataset=dataset,
        invoke_fn=invoke_fn,
        cache=cache,
    )
    results = selector.select_best()
    elapsed = time.time() - start

    print(results)
    print(f"LLM calls made: {call_count}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Cache now has {len(cache)} entries")

    # Show what's in the cache file.
    if os.path.exists(CACHE_PATH):
        print(f"\n--- Cache contents ({CACHE_PATH}) ---")
        with open(CACHE_PATH) as f:
            data = json.load(f)
        for key, entry in data.items():
            tokens = entry.get("tokens", {})
            tok_str = (
                ", ".join(f"{m}: {v[0]}/{v[1]}" for m, v in tokens.items())
                if tokens
                else "0/0"
            )
            print(
                f"  {key}: model={entry['model_name']}, score={entry['score']}, "
                f"latency={entry['latency']:.3f}s, tokens=[{tok_str}], "
                f"response={entry['response']!r}"
            )


if __name__ == "__main__":
    main()
