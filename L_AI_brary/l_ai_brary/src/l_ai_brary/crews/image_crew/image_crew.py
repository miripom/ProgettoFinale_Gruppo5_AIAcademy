from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from l_ai_brary.tools.image_tool import ImageGenerationTool

@CrewBase
class ImageCrew():
    """A CrewAI crew for image generation tasks.
    
    This crew manages a sequential workflow for extracting scenes from text,
    generating appropriate image prompts, and creating images based on those prompts.
    
    Attributes:
        agents: List of BaseAgent objects that perform different tasks in the workflow.
        tasks: List of Task objects that define the sequential operations.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def scene_extractor(self) -> Agent:
        """Create an agent responsible for extracting visual scenes from text.
        
        This agent analyzes text content to identify and extract descriptive
        scenes that can be visualized as images.
        
        Returns:
            Agent: A configured CrewAI agent for scene extraction tasks.
        """
        return Agent(
            config=self.agents_config['scene_extractor'], # type: ignore[index]
            verbose=True
        )

    @agent
    def image_prompt_generator(self) -> Agent:
        """Create an agent responsible for generating image prompts.
        
        This agent transforms extracted scenes into detailed, optimized prompts
        suitable for image generation models.
        
        Returns:
            Agent: A configured CrewAI agent for image prompt generation tasks.
        """
        return Agent(
            config=self.agents_config['image_prompt_generator'], # type:ignore
            verbose=True
        )

    @agent
    def image_creator(self) -> Agent:
        """Create an agent responsible for generating images.
        
        This agent uses the ImageGenerationTool to create actual images
        based on the optimized prompts provided by the prompt generator.
        
        Returns:
            Agent: A configured CrewAI agent equipped with image generation tools.
        """
        return Agent(
            config=self.agents_config['image_creator'], # type:ignore
            verbose=True,
            tools=[ImageGenerationTool()]
        )

    @task
    def scene_extraction_task(self) -> Task:
        """Create a task for extracting visual scenes from text content.
        
        This task configures the scene extraction process, defining how the
        scene_extractor agent should analyze text to identify visualizable content.
        
        Returns:
            Task: A configured CrewAI task for scene extraction.
        """
        return Task(
            config=self.tasks_config['scene_extraction_task'], # type:ignore
        )

    @task
    def image_prompt_task(self) -> Task:
        """Create a task for generating optimized image prompts.
        
        This task configures the prompt generation process, defining how the
        image_prompt_generator agent should transform scenes into detailed prompts.
        
        Returns:
            Task: A configured CrewAI task for image prompt generation.
        """
        return Task(
            config=self.tasks_config['image_prompt_task'] # type:ignore
        )
        
    @task
    def image_creation_task(self) -> Task:
        """Create a task for generating images from prompts.
        
        This task configures the image creation process, defining how the
        image_creator agent should use prompts to generate actual images.
        
        Returns:
            Task: A configured CrewAI task for image creation.
        """
        return Task(
            config=self.tasks_config['image_creation_task'] # type:ignore
        )
    @crew
    def crew(self) -> Crew:
        """Create and configure the complete ImageCrew workflow.
        
        This method assembles all agents and tasks into a sequential workflow
        for the complete image generation process: scene extraction → prompt 
        generation → image creation.
        
        Returns:
            Crew: A fully configured CrewAI crew ready to execute the image
                generation workflow.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
