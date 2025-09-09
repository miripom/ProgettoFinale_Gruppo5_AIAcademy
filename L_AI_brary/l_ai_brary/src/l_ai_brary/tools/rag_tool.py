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
    """
    Performs a hybrid semantic + keyword search over a Qdrant collection.

    Args:
        query (str): The natural language query or search string to look up.
        collection_name (str): The name of the Qdrant collection where the search
            should be performed.
        final_k (int, optional): The maximum number of top results to return.
            Defaults to 6.

    Returns:
        str: A formatted string of the retrieved documents suitable for passing
        into a prompt, or an explanatory error message if no results are found
        or an exception occurs.

    Usage Notes:
        - Always provide a valid `collection_name` that exists in Qdrant.
        - Use when you want the most relevant passages from the vector store
          to ground a response.
    """
    try:
        
        s = RAG_Settings()
        s.final_k = final_k
        s.collection = collection_name

        embeddings = get_embeddings(s)
        client = get_qdrant_client(s)

        hits = hybrid_search(client, s, query, embeddings)

        if not hits:
            return "Nessun risultato trovato nel RAG."
        return format_docs_for_prompt(hits)

    except Exception as e:
        return f"Errore durante la ricerca nel RAG: {str(e)}"
    
@tool("ListCollections")
def ListQdrantCollectionsTool() -> str:
    """
    Lists all available Qdrant collections.

    Args:
        None

    Returns:
        str: A newline-separated list of collection names available in Qdrant,
        or a message indicating no collections exist, or an error string if
        retrieval fails.

    Usage Notes:
        - Useful to discover valid collection names before calling HybridSearchTool.
        - Call this first if you are unsure of what collections are available.
    """
    try:
        s = RAG_Settings()
        client = get_qdrant_client(s)
        collections = client.get_collections()
        if not collections:
            return "Nessuna collezione trovata nel RAG."
        collections = [coll.name for coll in collections.collections]
        return "\n".join(collections)
    except Exception as e:
        return f"Errore durante il recupero delle collezioni nel RAG: {str(e)}"