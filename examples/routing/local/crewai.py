"""
Example: per-call random routing with a CrewAI crew.

The researcher+writer crew is unchanged; ``with LLMTracker(router=...)``
overrides the model per LLM call.  Each agent's LLM hits inside a
single ``agent.run()`` may end up on different models.

Prerequisites:
    1. ``pip install agentopt-py crewai python-dotenv``
    2. ``export OPENAI_API_KEY=...``

Usage:
    python examples/routing/local/crewai.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from crewai import LLM, Agent, Crew, Task

from agentopt import LLMTracker, RandomRouter


class MyAgent:
    """CrewAI researcher+writer crew. Nominal model is overridden per call."""

    def __init__(self) -> None:
        self.researcher_llm = LLM(model="gpt-4o-mini")
        self.writer_llm = LLM(model="gpt-4o-mini")

    def run(self, input_data: str) -> str:
        researcher = Agent(
            role="Researcher",
            goal="Research the topic and provide accurate information",
            backstory="You are a knowledgeable researcher.",
            llm=self.researcher_llm,
        )
        writer = Agent(
            role="Writer",
            goal="Write a concise answer based on research",
            backstory="You are a skilled writer who distills information.",
            llm=self.writer_llm,
        )
        research_task = Task(
            description=f"Research this question: {input_data}",
            expected_output="Factual information about the topic",
            agent=researcher,
        )
        write_task = Task(
            description=f"Write a concise answer to: {input_data}",
            expected_output="A clear, concise answer",
            agent=writer,
        )
        crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
        return str(crew.kickoff())


QUESTIONS = [
    "What is 7 * 8?",
    "What is the capital of France?",
    "What color is the sky on a clear day?",
]


if __name__ == "__main__":
    agent = MyAgent()
    router = RandomRouter(candidates=["gpt-4o-mini", "gpt-4.1-nano"], seed=0)
    with LLMTracker(router=router) as tracker:
        for i, q in enumerate(QUESTIONS, 1):
            with tracker.track(data_id=f"q{i}"):
                print(agent.run(q))
    tracker.print_summary()
