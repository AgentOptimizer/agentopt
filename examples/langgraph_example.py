"""
Example: LangGraph agent with agentopt.

Prerequisites:
    1. pip install langchain-openai langgraph agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agentopt import ModelSelector


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    answer: str


# ---------------------------------------------------------------------------
# Step 1: Define your agent class.
# __init__(models) receives a dict like {"planner": "gpt-4o", "solver": "gpt-4o-mini"}.
# run(input_data) runs the agent on a single datapoint and returns the output.
# ---------------------------------------------------------------------------

class MyAgent:
    """LangGraph planner+solver agent."""

    def __init__(self, models):
        planner_llm = ChatOpenAI(model=models["planner"])
        solver_llm = ChatOpenAI(model=models["solver"])

        def planner_node(state: AgentState) -> dict:
            response = planner_llm.invoke(
                [{"role": "system", "content": "Create a brief plan to answer the question."}]
                + state["messages"]
            )
            return {"plan": response.content}

        def solver_node(state: AgentState) -> dict:
            response = solver_llm.invoke(
                [
                    {"role": "system", "content": f"Follow this plan and answer concisely:\n{state['plan']}"},
                    state["messages"][-1],
                ]
            )
            return {"answer": response.content}

        graph = StateGraph(AgentState)
        graph.add_node("planner", planner_node)
        graph.add_node("solver", solver_node)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "solver")
        graph.add_edge("solver", END)
        self._app = graph.compile()

    def run(self, input_data):
        result = self._app.invoke({"messages": [{"role": "user", "content": input_data}]})
        return result["answer"]


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
