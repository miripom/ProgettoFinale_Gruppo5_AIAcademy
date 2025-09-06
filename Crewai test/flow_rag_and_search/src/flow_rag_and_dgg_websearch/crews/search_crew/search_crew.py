from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from crewai_tools import SerperDevTool
from crewai_tools import PatronusEvalTool


@CrewBase
class SearchCrew:
    """Crew che gestisce la ricerca sul web"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def search_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["search_agent"],  # type: ignore
            tools=[SerperDevTool(), PatronusEvalTool()]
        )

    @task
    def search_task(self) -> Task:
        return Task(config=self.tasks_config["search_task"])  # type: ignore

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
