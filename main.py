"""
Main file for model selection.
"""
from agent import MyLangchainAgent
from model_selection import ModelSelector
from typing import Optional, Callable
import pandas as pd
import matplotlib.pyplot as plt


def accuracy_fn(expected_answer: str, actual_output: str) -> bool:
    """
    Check if the expected answer appears in the actual output.
    
    Args:
        expected_answer: The expected answer string
        actual_output: The actual output from the agent
    
    Returns:
        True if expected_answer appears in actual_output (case-insensitive), False otherwise
    """
    if not expected_answer:
        return bool(actual_output.strip())
    
    # Check if expected answer appears anywhere in the output (case-insensitive)
    return expected_answer.lower() in actual_output.lower()


# Basic usage
dataset_dir: Optional[str] = "datasets"  # Path to evaluation dataset directory

# Create agent
agent = MyLangchainAgent()

# Command-level Model Selection
# You can use string model names - they'll be automatically converted to model objects
selector = ModelSelector(
    agent=agent,
    dataset_dir=dataset_dir,  # Optional: List of tasks/episodes
    models={
        "._model": [  # Attribute path to bind models to (agent._model)
            # String names are automatically converted to model objects
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-3-flash-preview",
        ]
    },
    accuracy_fn=accuracy_fn,
)

# Run model selection
print("Starting model selection...\n")
results_df = selector.select_best()

# Display results dataframe
print(f"\n{'='*60}")
print("Model Selection Results DataFrame:")
print(f"{'='*60}")
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv("model_selection_results.csv", index=False)
print(f"\nResults saved to model_selection_results.csv")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(results_df['latency_seconds'], results_df['accuracy'])

# Add labels for each point
for _, row in results_df.iterrows():
    plt.annotate(
        row['model_name'],
        (row['latency_seconds'], row['accuracy'])
    )

plt.xlabel('Latency (seconds)')
plt.ylabel('Accuracy')
plt.title('Model Performance: Accuracy vs Latency')
plt.grid(True)

# Save plot
plt.savefig('model_selection_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to model_selection_results.png")
plt.show()
