from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from l_ai_brary.tools.image_tool import ImageGenerationTool

@CrewBase
class ImageCrew():
    """Imagecrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def scene_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['scene_extractor'], # type: ignore[index]
            verbose=True
        )

    @agent
    def image_prompt_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['image_prompt_generator'], # type:ignore
            verbose=True
        )

    @agent
    def image_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['image_creator'], # type:ignore
            verbose=True,
            tools=[ImageGenerationTool()]
        )

    @task
    def scene_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['scene_extraction_task'], # type:ignore
        )

    @task
    def image_prompt_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_prompt_task'] # type:ignore
        )
        
    @task
    def image_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_creation_task'] # type:ignore
        )
    @crew
    def crew(self) -> Crew:
        """Creates the Imagecrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
