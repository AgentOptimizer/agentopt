"""
Example: CrewAI agent with agentopt.

Prerequisites:
    1. pip install crewai agentopt
    2. Set OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Crew, LLM, Task

from agentopt import ModelSelector


class MyAgent:
    """CrewAI crew with researcher + writer agents."""

    def __init__(self, models):
        self.researcher_llm = LLM(model=models["researcher"])
        self.writer_llm = LLM(model=models["writer"])

    def run(self, input_data):
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
        result = crew.kickoff()
        return str(result)


def eval_fn(expected, actual):
    return 1.0 if expected.lower() in str(actual).lower() else 0.0


dataset = [
    # Easy – every combo should get these
    ("What is 7 * 8?", "56"),
    ("What is the derivative of x^3?", "3x^2"),
    # Medium
    ("What is the integral of 1/(1+x^2) dx?", "arctan"),
    ("If log base 2 of x equals 5, what is x?", "32"),
    # Hard – weaker combos likely fail
    (
        "What is the sum of the series 1/1! + 1/2! + 1/3! + ... + 1/10! "
        "rounded to 6 decimal places?",
        "1.718282",
    ),
    (
        "A bag has 5 red and 3 blue balls. Two are drawn without replacement. "
        "What is the probability both are red? Give the fraction.",
        "5/14",
    ),
    ("Find the remainder when 2^100 is divided by 7.", "2",),
    ("What is the determinant of the matrix [[1,2,3],[4,5,6],[7,8,9]]?", "0",),
]


if __name__ == "__main__":
    selector = ModelSelector(
        agent=MyAgent,
        models={
            "researcher": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
            "writer": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],
        },
        eval_fn=eval_fn,
        dataset=dataset,
        method="brute_force",
    )

    results = selector.select_best(parallel=True)
    results.print_summary()

    best = results.get_best_combo()
    if best:
        print(f"\nBest combination: {best}")
