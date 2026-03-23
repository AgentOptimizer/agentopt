"""
Example: AG2 agent with agentopt.

Prerequisites:
    1. pip install ag2 agentopt
    2. Set OPENAI_API_KEY environment variable

Note: AG2's run() API spawns a background thread which breaks context
propagation. This example uses initiate_chat() which blocks in the
caller's thread and is required for agentopt.proxy compatibility.
"""

from dotenv import load_dotenv

load_dotenv()

import autogen

from agentopt import ModelSelector


# ---------------------------------------------------------------------------
# Step 1: Define your agent class.
# __init__(models) receives a dict like {"planner": "gpt-4o", "solver": "gpt-4o-mini"}.
# run(input_data) runs the agent on a single datapoint and returns the output.
# ---------------------------------------------------------------------------


class MyAgent:
    """AG2 planner+solver agent pair."""

    def __init__(self, models):
        self.planner = autogen.AssistantAgent(
            name="Planner",
            system_message="You are a planning assistant. Create a brief plan to answer the question. End your response with PLAN_DONE.",
            llm_config={"model": models["planner"]},
        )
        self.solver = autogen.AssistantAgent(
            name="Solver",
            system_message="You are a solver. Given a plan, produce a concise final answer. End your response with TERMINATE.",
            llm_config={"model": models["solver"]},
        )
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=4,
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
            code_execution_config=False,
        )

    def run(self, input_data):
        # NOTE: use initiate_chat(), not run().
        # run() spawns a background thread that breaks contextvar propagation.
        chat_result = self.user_proxy.initiate_chat(
            self.planner, message=f"Answer this question: {input_data}", max_turns=4,
        )

        # Extract the last non-empty assistant message as the answer
        for msg in reversed(chat_result.chat_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"].replace("TERMINATE", "").strip()
        return ""


# ---------------------------------------------------------------------------
# Step 2: Evaluation dataset — (input_data, expected_output) pairs.
# ---------------------------------------------------------------------------

dataset = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
]


# ---------------------------------------------------------------------------
# Step 3: Evaluation function — score agent output against expected answer.
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


# ---------------------------------------------------------------------------
# Step 4: Run model selection.
# Two steps ("planner", "solver") × 3 models = 9 combinations.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    selector = ModelSelector(
        agent=MyAgent,
        models={
            "planner": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
            "solver": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        },
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",  # or "auto" for smarter selection algorithms
    )

    results = selector.select_best(parallel=True)
    results.print_summary()
    results.plot_pareto()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
