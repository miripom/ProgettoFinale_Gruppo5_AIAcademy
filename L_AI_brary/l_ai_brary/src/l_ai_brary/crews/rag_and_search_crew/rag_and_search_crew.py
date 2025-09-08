from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from l_ai_brary.tools.rag_tool import HybridSearchTool, ListQdrantCollectionsTool
from crewai_tools import SerperDevTool

@CrewBase
class RagAndSearchCrew:
    """Crew che gestisce query contro la knowledge base locale (RAG)"""

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],  # type: ignore
            tools=[HybridSearchTool, ListQdrantCollectionsTool]   # tool generico per interrogare il RAG
        )
    
    @agent
    def search_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["search_agent"],  # type: ignore
            tools=[SerperDevTool()]
        )
    
    @agent
    def synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["synthesizer_agent"],  # type: ignore
            tools=[],
        )


    @task
    def rag_task(self) -> Task:
        return Task(config=self.tasks_config["rag_task"])  # type: ignore
    
    @task
    def search_task(self) -> Task:
        return Task(config=self.tasks_config["search_task"]) # type:ignore

    @task
    def synthesizer_task(self) -> Task:
        return Task(
            config=self.tasks_config["synthesizer_task"],  # type: ignore
            context=[self.rag_task, self.search_task],
        )
    

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
