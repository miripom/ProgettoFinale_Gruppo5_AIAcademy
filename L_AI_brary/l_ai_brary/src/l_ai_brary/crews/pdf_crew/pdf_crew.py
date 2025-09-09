from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class PdfCrew():
    """A CrewAI crew for PDF processing and analysis tasks.
    
    This crew manages a sequential workflow for researching content from PDFs
    and generating comprehensive reports based on the extracted information.
    
    Attributes:
        agents: List of BaseAgent objects that perform research and reporting tasks.
        tasks: List of Task objects that define the sequential operations.
    """

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def researcher(self) -> Agent:
        """Create an agent responsible for researching and extracting information from PDFs.
        
        This agent analyzes PDF documents to extract relevant information,
        identify key insights, and gather data for further processing.
        
        Returns:
            Agent: A configured CrewAI agent for research tasks on PDF content.
        """
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        """Create an agent responsible for analyzing research data and generating reports.
        
        This agent takes the information gathered by the researcher and transforms
        it into structured, comprehensive reports with analysis and insights.
        
        Returns:
            Agent: A configured CrewAI agent for report generation and analysis tasks.
        """
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        """Create a task for researching and extracting information from PDF documents.
        
        This task configures the research process, defining how the researcher
        agent should analyze PDF content and extract relevant information.
        
        Returns:
            Task: A configured CrewAI task for PDF research operations.
        """
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        """Create a task for generating comprehensive reports from research data.
        
        This task configures the report generation process, defining how the
        reporting_analyst agent should transform research findings into structured
        reports. The output is automatically saved to a markdown file.
        
        Returns:
            Task: A configured CrewAI task for report generation with file output.
        """
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create and configure the complete PdfCrew workflow.
        
        This method assembles all agents and tasks into a sequential workflow
        for the complete PDF processing pipeline: research → analysis → reporting.
        The crew processes PDF documents and generates comprehensive markdown reports.
        
        Returns:
            Crew: A fully configured CrewAI crew ready to execute the PDF
                processing and reporting workflow.
        
        Note:
            The crew uses sequential processing by default. For hierarchical
            processing, uncomment the hierarchical process line in the configuration.
        """
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
