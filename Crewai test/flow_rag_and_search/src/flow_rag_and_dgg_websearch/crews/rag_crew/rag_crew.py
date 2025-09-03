from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from flow_rag_and_dgg_websearch.tools.RAG_tool import RagTool

@CrewBase
class RagCrew:
    """Crew che gestisce query contro la knowledge base locale (RAG)"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],  # type: ignore
            tools=[RagTool]   # tool generico per interrogare il RAG
        )

    @task
    def rag_task(self) -> Task:
        return Task(config=self.tasks_config["rag_task"])  # type: ignore

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
