import os
from dotenv import load_dotenv
from crewai.tools import tool
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

@tool("RAGRetriever")
def RagTool(query: str, k: int = 3) -> str:
    """
    Recupera i documenti più rilevanti dalla knowledge base locale tramite FAISS.
    Usa questo tool quando vuoi rispondere a una domanda sfruttando i documenti del RAG.

    query: str = La domanda a cui rispondere interrogando la knowledge base locale
    k: int = Numero di documenti più rilevanti da recuperare (default: 3)
    """
    try:
        docs = []
        saved_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if saved_base:
            del os.environ["AZURE_OPENAI_ENDPOINT"]
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_deployment=os.environ["DEPLOYMENT_EMBEDDING"],
            azure_endpoint=saved_base,
            api_version=os.environ["OPENAI_API_VERSION"],
            )
        if saved_base:
            os.environ["AZURE_OPENAI_ENDPOINT"] = saved_base

        
        faiss_0__or__qdrant_1 = int(os.environ["FAISS_0__OR__QDRANT_1"])
        if faiss_0__or__qdrant_1 == 0:

            # Path all'indice salvato in precedenza
            path_to_indices_folder = "C:/Users/LH668YN/OneDrive - EY/Desktop/CrewAI/3_Flow_Rag_and_DDG_WebSearch/Local_RAG/indices"

            # Caricamento FAISS
            vector_store = FAISS.load_local(
                folder_path=path_to_indices_folder,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            # Similarity search
            docs: List[Document] = vector_store.similarity_search(query, k=k)


        elif faiss_0__or__qdrant_1 == 1:

            # Caricamento Qdrant
            client = QdrantClient(host="localhost", port=6333)
            rag_index = Qdrant(
                                client=client,
                                collection_name="index_risposte-sbagliate",
                                embeddings=embeddings,
            )        
            # Similarity search
            docs: List[Document] = rag_index.similarity_search(query, k=k)


        if not docs:
            return "Nessun documento rilevante trovato nella knowledge base."

        # Formatta i risultati
        formatted = []
        for i, d in enumerate(docs, 1):
            snippet = d.page_content.strip().replace("\n", " ")
            meta = d.metadata if d.metadata else {}
            formatted.append(f"[{i}] {snippet} (source: {meta.get('source', 'N/A')})")

        return "\n".join(formatted)

    except Exception as e:
        return f"Errore durante la ricerca nel RAG: {str(e)}"
