from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from l_ai_brary.tools.sanitize_tool import LexicalTool
from typing import List


@CrewBase
class SanitizeCrew():
    """A CrewAI crew for text sanitization and content filtering tasks.
    
    This crew manages a sequential workflow for cleaning and filtering text content
    through both lexical and semantic analysis. The process involves lexical cleaning
    followed by semantic filtering to ensure content quality and appropriateness.
    
    Attributes:
        agents: List of BaseAgent objects that perform lexical and semantic sanitization.
        tasks: List of Task objects that define the sequential cleaning operations.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def lexical_sanitizer(self) -> Agent:
        """Create an agent responsible for lexical text sanitization.
        
        This agent performs lexical-level cleaning of text content, including
        grammar correction, spelling fixes, formatting normalization, and other
        word-level sanitization tasks using specialized lexical tools.
        
        Returns:
            Agent: A configured CrewAI agent equipped with LexicalTool for
                text cleaning and normalization operations.
        """
        return Agent(
            config=self.agents_config['lexical_sanitizer'], # type:ignore
            verbose=True,
            tools=[LexicalTool()]
        )

    @agent
    def semantic_sanitizer(self) -> Agent:
        """Create an agent responsible for semantic content filtering and validation.
        
        This agent performs semantic-level analysis and filtering of text content,
        evaluating meaning, context appropriateness, content quality, and ensuring
        the text meets semantic standards and guidelines.
        
        Returns:
            Agent: A configured CrewAI agent for semantic content analysis and filtering.
        """
        return Agent(
            config=self.agents_config['semantic_sanitizer'], # type:ignore
            verbose=True,
        )

    @task
    def lexical_cleaning(self) -> Task:
        """Create a task for lexical-level text cleaning and normalization.
        
        This task configures the lexical cleaning process, defining how the
        lexical_sanitizer agent should process text to fix grammar, spelling,
        formatting, and other word-level issues.
        
        Returns:
            Task: A configured CrewAI task for lexical text sanitization.
        """
        return Task(
            config=self.tasks_config['lexical_cleaning'], # type:ignore
        )

    @task
    def semantic_filtering(self) -> Task:
        """Create a task for semantic content filtering and quality validation.
        
        This task configures the semantic filtering process, defining how the
        semantic_sanitizer agent should analyze content meaning, context, and
        appropriateness to ensure high-quality, suitable text output.
        
        Returns:
            Task: A configured CrewAI task for semantic content filtering.
        """
        return Task(
            config=self.tasks_config['semantic_filtering'] # type:ignore
        )

    @crew
    def crew(self) -> Crew:
        """Create and configure the complete SanitizeCrew workflow.
        
        This method assembles all agents and tasks into a sequential workflow
        for comprehensive text sanitization: lexical cleaning → semantic filtering.
        The process ensures both technical correctness and content appropriateness.
        
        Returns:
            Crew: A fully configured CrewAI crew ready to execute the complete
                text sanitization and filtering workflow.
        """

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential
        )
