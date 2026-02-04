"""
CrewAI wrappers for model selection compatibility.

Provides OptCrewAgent (single-agent) and OptCrew (multi-agent) wrappers
that are compatible with ModelSelector's invoke interface.
"""

from typing import Any, Dict, List

from crewai import Agent, Task, Crew, Process, LLM

from agentopt import ModelProxy
from agentopt.types import AgentProtocol


class OptCrewAgent(AgentProtocol):
    """
    Single-agent wrapper compatible with ModelSelector.

    Usage:
        agent = OptCrewAgent(
            model="openai/gpt-4o-mini",
            role="Calculator",
            goal="Perform calculations",
            backstory="You are a math expert",
            task_description="Answer the question",
            expected_output="A number"
        )

        selector = ModelSelector(
            agent=agent,
            models={"._model": ["openai/gpt-4o-mini", "openai/gpt-4o"]},
            accuracy_fn=check_accuracy,
        )
    """

    def __init__(
        self,
        model: str,
        role: str,
        goal: str,
        backstory: str,
        task_description: str,
        expected_output: str,
        verbose: bool = False,
    ):
        self._model = ModelProxy("_model", LLM(model=model))
        self._role = role
        self._goal = goal
        self._backstory = backstory
        self._task_description = task_description
        self._expected_output = expected_output
        self._verbose = verbose

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ModelSelector calls this with {"messages": [...]}."""
        messages = input_dict.get("messages", [])
        question = (
            messages[0].get("content", self._task_description)
            if messages
            else self._task_description
        )

        llm = LLM(model=self._model._model)

        agent = Agent(
            role=self._role,
            goal=self._goal,
            backstory=self._backstory,
            llm=llm,
            verbose=self._verbose,
        )
        task = Task(
            description=question, expected_output=self._expected_output, agent=agent
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self._verbose,
        )

        result = crew.kickoff()
        return {"messages": [{"role": "assistant", "content": str(result)}]}


class OptCrew(AgentProtocol):
    """
    Multi-agent crew wrapper compatible with ModelSelector.

    Usage:
        crew = OptCrew(
            agent_configs=[
                {"role": "Researcher", "goal": "Research", "backstory": "Expert", "model": "openai/gpt-4o-mini"},
                {"role": "Writer", "goal": "Write", "backstory": "Writer", "model": "openai/gpt-4o"},
            ],
            task_configs=[
                {"description": "Research AI", "expected_output": "Summary"},
                {"description": "Write blog", "expected_output": "Blog post"},
            ],
        )

        selector = ModelSelector(
            agent=crew,
            models={"._models.0": ["openai/gpt-4o-mini", "openai/gpt-4o"]},
            accuracy_fn=check_accuracy,
        )
    """

    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        task_configs: List[Dict[str, Any]],
        process: Process = Process.sequential,
        verbose: bool = False,
    ):
        self._agent_configs = agent_configs
        self._task_configs = task_configs
        self._process = process
        self._verbose = verbose

        self._models = []
        for config in agent_configs:
            llm = LLM(model=config["model"])
            self._models.append(ModelProxy(f"_models.{len(self._models)}", llm))

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ModelSelector calls this with {"messages": [...]}."""
        messages = input_dict.get("messages", [])
        override_question = messages[0].get("content") if messages else None

        # Create agents with current LLMs
        agents = []
        for i, config in enumerate(self._agent_configs):
            raw_model = self._models[i]._model
            llm = LLM(model=raw_model)
            agents.append(
                Agent(
                    role=config["role"],
                    goal=config["goal"],
                    backstory=config["backstory"],
                    llm=llm,
                    verbose=config.get("verbose", self._verbose),
                )
            )

        # Create tasks
        tasks = []
        for i, task_config in enumerate(self._task_configs):
            agent_index = task_config.get("agent_index", min(i, len(agents) - 1))
            description = (
                override_question
                if i == 0 and override_question
                else task_config["description"]
            )
            tasks.append(
                Task(
                    description=description,
                    expected_output=task_config["expected_output"],
                    agent=agents[agent_index],
                )
            )

        crew = Crew(
            agents=agents, tasks=tasks, process=self._process, verbose=self._verbose
        )
        result = crew.kickoff()
        return {"messages": [{"role": "assistant", "content": str(result)}]}
