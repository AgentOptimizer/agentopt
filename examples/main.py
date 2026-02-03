"""
Unified model selection for LangChain and CrewAI agents.
"""

from agentopt import ModelSelector
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


def accuracy_fn(expected_answer: str, actual_output: str) -> bool:
    """Check if the expected answer appears in the actual output."""
    if not expected_answer:
        return bool(actual_output.strip())
    return expected_answer.lower() in actual_output.lower()


# =============================================================================
# Choose which agent to use
# =============================================================================

USE_CREWAI = True  # Set to False for LangChain

if USE_CREWAI:
    from agentopt import OptCrewAgent

    agent = OptCrewAgent(
        model="openai/gpt-4o-mini",
        role="Calculator",
        goal="Perform calculations accurately",
        backstory="You are a math expert",
        task_description="Answer the math question",
        expected_output="A number",
    )
    model_path = "._model"
else:
    from agentopt import OptLangchainAgent
    from langchain_core.tools import tool

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    agent = OptLangchainAgent(
        model="openai/gpt-4o-mini",
        tools=[add],
        system_prompt="You are a helpful math assistant. Use the add tool when needed.",
    )
    model_path = "._model"


# =============================================================================
# Model Selection - Same API for both agent types
# =============================================================================

dataset_dir: Optional[str] = "examples/datasets"

selector = ModelSelector(
    agent=agent,
    dataset_dir=dataset_dir,
    models={
        model_path: [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
        ]
    },
    accuracy_fn=accuracy_fn,
)

print("Starting model selection...\n")
results_df = selector.select_best()

print(f"\n{'='*60}")
print("Model Selection Results DataFrame:")
print(f"{'='*60}")
print(results_df.to_string(index=False))

results_df.to_csv("examples/model_selection_results.csv", index=False)
print(f"\nResults saved to examples/model_selection_results.csv")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(results_df["latency_seconds"], results_df["accuracy"])

for _, row in results_df.iterrows():
    plt.annotate(row["model_name"], (row["latency_seconds"], row["accuracy"]))

plt.xlabel("Latency (seconds)")
plt.ylabel("Accuracy")
plt.title("Model Performance: Accuracy vs Latency")
plt.grid(True)

plt.savefig("examples/model_selection_results.png", dpi=300, bbox_inches="tight")
print("Plot saved to examples/model_selection_results.png")
plt.show()
