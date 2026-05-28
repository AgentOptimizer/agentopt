"""
Example: LlamaIndex agent with agentopt.

Prerequisites:
    1. pip install llama-index-core llama-index-llms-openai agentopt-py
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from agentopt import ModelSelector


# Tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a and returns the result"""
    return a - b


def divide(a: float, b: float) -> float:
    """Divide a by b and returns the result"""
    if b == 0:
        return float("inf")
    return a / b


# ---------------------------------------------------------------------------
# Step 1: Define your agent class.
# __init__(models) receives a dict like {"agent": "gpt-4o-mini"}.
# run(input_data) runs the agent on a single datapoint and returns the output.
# Note: run() can be async — AgentOpt detects this automatically.
# ---------------------------------------------------------------------------


class MyAgent:
    """LlamaIndex math agent with calculator tools."""

    def __init__(self, models):
        llm = LlamaOpenAI(model=models["agent"])

        agent = FunctionAgent(
            name="MathAgent",
            description="Solves math problems using calculator tools",
            tools=[multiply, add, subtract, divide],
            llm=llm,
            system_prompt=(
                "You are a helpful assistant that can perform mathematical operations. "
                "When asked to calculate something, use the available tools to compute the result."
            ),
        )

        self._workflow = AgentWorkflow(agents=[agent], root_agent="MathAgent")

    async def run(self, input_data):
        response = await self._workflow.run(user_msg=input_data)
        return str(response)


# ---------------------------------------------------------------------------
# Step 2: Evaluation dataset — (input_data, expected_output) pairs.
# ---------------------------------------------------------------------------

dataset = [
    ("What is 2 + 2?", "4"),
    ("What is 5 * 3?", "15"),
    ("What is 10 - 4?", "6"),
]


# ---------------------------------------------------------------------------
# Step 3: Evaluation function — score agent output against expected answer.
# ---------------------------------------------------------------------------


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


# ---------------------------------------------------------------------------
# Step 4: Run model selection.
# Single step ("agent") × 3 models = 3 combinations.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    selector = ModelSelector(
        agent=MyAgent,
        models={"agent": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],},
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",  # or "auto" for smarter selection algorithms
        objective_mode="pareto",
    )

    results = selector.select_best(parallel=True)
    results.print_summary()
    results.plot_pareto()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
