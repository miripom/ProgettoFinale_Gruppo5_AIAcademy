from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from l_ai_brary.tools.rag_tool import HybridSearchTool, ListQdrantCollectionsTool
from crewai_tools import SerperDevTool

@CrewBase
class RagAndSearchCrew():
    """A CrewAI crew for hybrid RAG and web search operations.
    
    This crew combines local knowledge base queries (RAG) with web search capabilities
    to provide comprehensive information retrieval and synthesis. The workflow includes
    querying local collections, performing web searches, and synthesizing results.
    
    Attributes:
        agents: List of BaseAgent objects that perform RAG queries, web searches, and synthesis.
        tasks: List of Task objects that define the sequential operations.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_agent(self) -> Agent:
        """Create an agent responsible for querying local knowledge base using RAG.
        
        This agent performs hybrid searches against local Qdrant collections,
        combining semantic and keyword search to retrieve relevant information
        from the stored knowledge base.
        
        Returns:
            Agent: A configured CrewAI agent equipped with HybridSearchTool and
                ListQdrantCollectionsTool for RAG operations.
        """
        return Agent(
            config=self.agents_config["rag_agent"],  # type: ignore
            verbose=True,
            tools=[HybridSearchTool, ListQdrantCollectionsTool]   # tool generico per interrogare il RAG
        )
    
    @agent
    def search_agent(self) -> Agent:
        """Create an agent responsible for performing web searches.
        
        This agent conducts web searches using SerperDev to gather additional
        information that may not be available in the local knowledge base,
        providing up-to-date and comprehensive search results.
        
        Returns:
            Agent: A configured CrewAI agent equipped with SerperDevTool for web search operations.
        """
        return Agent(
            config=self.agents_config["search_agent"],  # type: ignore
            verbose=True,
            tools=[SerperDevTool()]
        )
    @agent
    def synthesizer_agent(self) -> Agent:
        """Create an agent responsible for synthesizing information from multiple sources.
        
        This agent combines and synthesizes information gathered from both the RAG
        system and web search results, creating coherent and comprehensive responses
        that leverage all available information sources.
        
        Returns:
            Agent: A configured CrewAI agent for information synthesis and integration.
        """
        return Agent(
            config=self.agents_config["synthesizer_agent"],  # type: ignore
            verbose=True,
            tools=[],
        )

    @task
    def rag_task(self) -> Task:
        """Create a task for querying the local knowledge base using RAG.
        
        This task configures the RAG query process, defining how the rag_agent
        should search through local collections to retrieve relevant information.
        
        Returns:
            Task: A configured CrewAI task for RAG-based information retrieval.
        """
        return Task(config=self.tasks_config["rag_task"])  # type: ignore
    
    @task
    def search_task(self) -> Task:
        """Create a task for performing web searches.
        
        This task configures the web search process, defining how the search_agent
        should conduct searches to gather additional information from the internet.
        
        Returns:
            Task: A configured CrewAI task for web search operations.
        """
        return Task(config=self.tasks_config["search_task"]) # type:ignore
    
    @task
    def synthesizer_task(self) -> Task:
        """Create a task for synthesizing information from RAG and search results.
        
        This task configures the synthesis process, defining how the synthesizer_agent
        should combine and integrate information from both local knowledge base
        queries and web search results into a coherent response.
        
        Returns:
            Task: A configured CrewAI task for information synthesis and integration.
        
        Note:
            Context dependencies can be configured to ensure this task receives
            outputs from both rag_task and search_task.
        """
        return Task(
            config=self.tasks_config["synthesizer_task"],  # type: ignore
            # context=[self.rag_task, self.search_task],
        )

    @crew
    def crew(self) -> Crew:
        """Create and configure the complete RagAndSearchCrew workflow.
        
        This method assembles all agents and tasks into a sequential workflow
        that combines local knowledge base queries with web search capabilities.
        The complete pipeline: RAG query → web search → information synthesis.
        
        Returns:
            Crew: A fully configured CrewAI crew ready to execute the hybrid
                RAG and search workflow for comprehensive information retrieval.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential
        )
