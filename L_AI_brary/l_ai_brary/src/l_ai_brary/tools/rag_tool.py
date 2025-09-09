"""RAG (Retrieval-Augmented Generation) Tools for CrewAI.

This module provides CrewAI tools for interacting with Qdrant vector databases
to perform hybrid search operations and collection management. The tools enable
agents to retrieve relevant information from indexed documents using both
semantic and keyword-based search approaches.

The module includes:
    - HybridSearchTool: Performs semantic + keyword search on Qdrant collections
    - ListQdrantCollectionsTool: Lists available collections in the vector database

Dependencies:
    - Qdrant vector database for document storage and retrieval
    - Azure OpenAI embeddings for semantic search
    - LangChain for document processing and vector store integration

Environment Setup:
    Requires proper configuration of RAG_Settings with Qdrant connection
    parameters and Azure OpenAI credentials for embeddings.
"""

import os
from crewai.tools import tool
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from l_ai_brary.utils.rag_utils import RAG_Settings, get_embeddings, get_qdrant_client, hybrid_search, format_docs_for_prompt

@tool("RAGRetriever")
def HybridSearchTool(query: str, collection_name: str, final_k: int = 6) -> str:
    """Performs hybrid semantic and keyword search over a Qdrant collection.

    This tool combines semantic similarity search using embeddings with keyword-based
    search to retrieve the most relevant documents from a Qdrant vector database.
    The hybrid approach provides better recall and precision compared to using either
    method alone.

    The search process:
    1. Generates embeddings for the input query using Azure OpenAI
    2. Performs semantic search using vector similarity
    3. Combines with keyword matching for comprehensive results
    4. Returns formatted document excerpts ready for prompt integration

    Args:
        query (str): The natural language query or search string to look up.
            Should be descriptive and specific for better results.
        collection_name (str): The name of the Qdrant collection where the search
            should be performed. Must be a valid existing collection.
        final_k (int, optional): The maximum number of top results to return.
            Defaults to 6. Higher values provide more context but may introduce noise.

    Returns:
        str: A formatted string containing the retrieved document excerpts,
            ready to be included in a prompt. Returns an error message if no
            results are found or if an exception occurs during the search.

    Examples:
        >>> HybridSearchTool("machine learning algorithms", "tech_docs", 5)
        "Document 1: Introduction to ML algorithms...\nDocument 2: ..."
        
    Note:
        Always ensure the collection_name exists before calling this tool.
        Use ListQdrantCollectionsTool() to discover available collections.
    """
    try:
        # Initialize RAG settings with search parameters
        s = RAG_Settings()
        s.final_k = final_k
        s.collection = collection_name

        # Get embeddings client and Qdrant client from settings
        embeddings = get_embeddings(s)
        client = get_qdrant_client(s)

        # Perform hybrid search combining semantic and keyword approaches
        hits = hybrid_search(client, s, query, embeddings)

        # Check if any results were found
        if not hits:
            return "Nessun risultato trovato nel RAG."
            
        # Format the retrieved documents for prompt integration
        return format_docs_for_prompt(hits)

    except Exception as e:
        return f"Errore durante la ricerca nel RAG: {str(e)}"
    
@tool("ListCollections")
def ListQdrantCollectionsTool() -> str:
    """Lists all available collections in the Qdrant vector database.

    This tool connects to the configured Qdrant instance and retrieves a list
    of all available collections. This is useful for discovering what document
    collections are available for search operations before using HybridSearchTool.

    The tool handles the complete workflow:
    1. Establishes connection to Qdrant using configured settings
    2. Queries the database for available collections
    3. Extracts collection names from the response
    4. Returns a formatted list of collection names

    Returns:
        str: A newline-separated list of collection names available in Qdrant.
            Returns a message indicating no collections exist if the database
            is empty, or an error string if the retrieval operation fails.

    Examples:
        >>> ListQdrantCollectionsTool()
        "documents\nbooks\npapers\nmanuals"
        
    Note:
        This tool should be called before HybridSearchTool when the available
        collection names are unknown. It requires properly configured Qdrant
        connection settings in RAG_Settings.
    """
    try:
        # Initialize RAG settings and get Qdrant client
        s = RAG_Settings()
        client = get_qdrant_client(s)
        
        # Query Qdrant for all available collections
        response = client.get_collections()

        # Check if any collections exist in the database
        if not response.collections:
            return "Nessuna collezione trovata nel RAG."

        # Extract collection names from the response
        names = [coll.name for coll in response.collections]
        
        # Return formatted list of collection names
        return "\n".join(names)

    except Exception as e:
        return f"Errore durante il recupero delle collezioni nel RAG: {str(e)}"