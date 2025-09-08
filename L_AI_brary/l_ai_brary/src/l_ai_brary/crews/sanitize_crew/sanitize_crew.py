from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from src.l_ai_brary.tools.sanitize_tool import LexicalTool
from typing import List


@CrewBase
class SanitizeCrew():
    """SanitizeCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def lexical_sanitizer(self) -> Agent:
        return Agent(
            config=self.agents_config['lexical_sanitizer'],
            verbose=True,
            tools=[LexicalTool()]
        )

    @agent
    def semantic_sanitizer(self) -> Agent:
        return Agent(
            config=self.agents_config['semantic_sanitizer'],
            verbose=True,
        )

    @task
    def lexical_cleaning(self) -> Task:
        return Task(
            config=self.tasks_config['lexical_cleaning'],
        )

    @task
    def semantic_filtering(self) -> Task:
        return Task(
            config=self.tasks_config['semantic_filtering']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SanitizeCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential
        )
