
"""RAG utilities for document processing, indexing, and retrieval.

This module provides comprehensive utilities for implementing a Retrieval-Augmented Generation (RAG)
system using Qdrant vector database. It includes functionality for:

- Document loading and chunking
- Vector embeddings and indexing in Qdrant
- Hybrid search combining semantic and text-based retrieval
- Maximum Marginal Relevance (MMR) for result diversification
- RAG chain building for question-answering

The module supports PDF documents and integrates with Azure OpenAI embeddings and LLMs
through LangChain. It implements a sophisticated hybrid search strategy that combines
semantic similarity search with keyword-based text search for improved retrieval quality.
"""
from __future__ import annotations

import os
import time
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Iterable, List, Tuple, cast
from dataclasses import dataclass
from l_ai_brary import main as m
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import init_chat_model
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    ScalarType,
    PointStruct,
    SearchParams,
    Filter,
    FieldCondition,
    MatchText,
)


load_dotenv()
@dataclass
class RAG_Settings:
    """Configuration settings for the RAG pipeline.
    
    This class contains all configurable parameters for the RAG system including
    Qdrant connection settings, embedding configuration, chunking parameters,
    search settings, and LLM configuration.
    
    Attributes:
        qdrant_url (str): URL for Qdrant vector database connection.
        collection (str): Name of the Qdrant collection to use.
        upsert_batch_size (int): Batch size for uploading vectors to Qdrant.
        emb_model_name (str): Name of the embedding model to use.
        chunk_size (int): Maximum size of text chunks in characters.
        chunk_overlap (int): Overlap between consecutive chunks in characters.
        top_n_semantic (int): Number of candidates from semantic search.
        top_n_text (int): Number of candidates from text search.
        final_k (int): Final number of results to return.
        alpha (float): Weight for semantic search in hybrid fusion (0-1).
        text_boost (float): Boost factor for text search matches.
        use_mmr (bool): Whether to use Maximum Marginal Relevance for diversification.
        mmr_lambda (float): MMR balance parameter between relevance and diversity.
        lm_base_env (str): Environment variable name for LLM base URL.
        lm_key_env (str): Environment variable name for LLM API key.
        lm_model_env (str): Environment variable name for LLM model name.
    """

    qdrant_url: str = "http://localhost:6333"        # Qdrant URL
    collection: str = "rag_chunks"                   # Collection name
    upsert_batch_size: int = 100                     # Upsert batch size
    emb_model_name: str = "text-embedding-ada-002"   # Embedding model
    chunk_size: int = 2000                           # Chunk size
    chunk_overlap: int = 100                         # Overlap size
    top_n_semantic: int = 30                         # Candidates for semantic search
    top_n_text: int = 100                            # Candidates for text search
    final_k: int = 6                                 # Final results count
    alpha: float = 0.75                              # Semantic weight
    text_boost: float = 0.20                         # Text boost
    use_mmr: bool = True                             # Use MMR diversification
    mmr_lambda: float = 0.6                          # MMR balance
    lm_base_env: str = "OPENAI_BASE_URL"             # LLM base URL env
    lm_key_env: str = "OPENAI_API_KEY"               # LLM API key env
    lm_model_env: str = "LMSTUDIO_MODEL"             # LLM model env



#################################################################### 
# Qdrant indexing functions
####################################################################



def index_pdf_in_qdrant(pdf_path: Path, rag_settings: RAG_Settings, crewai_flow: m.ChatbotFlow | None = None):
    """Index a PDF document in Qdrant vector database.
    
    This function processes a PDF document by loading, chunking, embedding, and storing
    it in a Qdrant collection. It creates a new collection named after the PDF file
    and populates it with vector embeddings of document chunks.
    
    Args:
        pdf_path (Path): Path to the PDF file to index.
        rag_settings (RAG_Settings): Configuration settings for the RAG pipeline.
        crewai_flow (ChatbotFlow | None, optional): Optional ChatbotFlow instance to 
            update with progress messages. Defaults to None.
    
    Note:
        This function will recreate the collection if it already exists, removing
        any previously indexed content for the same PDF.
    """
    rag_settings.collection = pdf_path.stem

    qdrant_client = get_qdrant_client(rag_settings)
    embeddings = get_embeddings(rag_settings)

    doc = load_document(pdf_path)
    print(f"✅ Loaded document '{pdf_path.name}' with {len(doc)} pages.\n")

    print(f"📄 Splitting document '{pdf_path.name}' into chunks...")
    chunks = split_documents(doc, rag_settings)
    print(f"✅ '{pdf_path.name}' split into {len(chunks)} chunks.\n")

    vector_size = len(embeddings.embed_query("hello world"))
    recreate_collection_for_rag(qdrant_client, rag_settings, vector_size)
    upsert_chunks(qdrant_client, rag_settings, chunks, embeddings)
    
    if crewai_flow:
        crewai_flow.state.chat_history.append({"role": "assistant", "content": f"✅ PDF **{pdf_path.stem}** fully analyzed and ready!"})


def get_qdrant_client(settings: RAG_Settings) -> QdrantClient:
    """Get a configured Qdrant client instance.
    
    Args:
        settings (RAG_Settings): Configuration settings containing Qdrant URL.
        
    Returns:
        QdrantClient: Configured Qdrant client with timeout settings.
    """
    return QdrantClient(url=settings.qdrant_url, timeout=60)

#################################################################### 
# RAGAS functions
####################################################################

def get_llm(settings: RAG_Settings):
    """Initialize and configure the Language Model for RAG.
    
    Attempts to initialize an LLM using environment variables specified in settings.
    Performs a test query to verify the LLM is working correctly.
    
    Args:
        settings (RAG_Settings): Configuration settings containing LLM environment 
            variable names.
    
    Returns:
        ChatOpenAI | None: Initialized LLM instance if successful, None otherwise.
        
    Note:
        Requires OPENAI_BASE_URL, OPENAI_API_KEY, and LMSTUDIO_MODEL environment
        variables to be set according to the settings configuration.
    """
    try:
        base = os.getenv(settings.lm_base_env)
        key = os.getenv(settings.lm_key_env)
        model_name = os.getenv(settings.lm_model_env)
        if not (base and key and model_name):
            print("LLM not configured")
            return None
        llm = init_chat_model(model_name, model_provider="azure_openai")
        test_response = llm.invoke("test")
        if test_response:
            print("LLM configured")
            return llm
        print("LLM test failed")
        return None
    except Exception as e:
        print(f"LLM error: {e}")
        return None
    
def build_rag_chain(llm):
    """Build a RAG chain for question-answering with source citations.
    
    Creates a LangChain pipeline that combines context and questions to generate
    answers with proper source attribution. The chain is configured to respond
    in Italian or English and emphasizes using only provided content.
    
    Args:
        llm: The language model instance to use for generation.
        
    Returns:
        Chain: A LangChain Runnable that takes context and question as input
            and returns a formatted answer with citations.
            
    Note:
        The system prompt instructs the model to cite sources and explicitly
        state when questions cannot be answered using the provided content.
    """
    system_prompt = (
        "Sei un assistente. Rispondi in italiano o inglese. Usa solo i contenuti forniti dallo user per rispondere alle domande, riformulandoli se necessario. "
        "Cita sempre le fonti. Sii chiaro e conciso. Solo se la domanda dello user non è in alcun modo rispondibile usando il contenuto fornito, dichiaralo esplicitamente"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Domanda:\n{question}\n\nCONTENUTO:\n{context}\n\nIstruzioni:\n1) Rispondi solo col contenuto.\n2) Cita fonti.\n3) Niente invenzioni.")
    ])
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_embeddings(settings: RAG_Settings) -> AzureOpenAIEmbeddings:
    """Get configured Azure OpenAI embeddings instance.
    
    Args:
        settings (RAG_Settings): Configuration settings containing embedding model name.
        
    Returns:
        AzureOpenAIEmbeddings: Configured embeddings instance for text vectorization.
    """
    saved_base = os.environ["AZURE_OPENAI_ENDPOINT"] or os.environ["AZURE_OPENAI_ENDPOINT"]
    if saved_base:
        del os.environ["AZURE_API_BASE"]
        del os.environ["AZURE_OPENAI_ENDPOINT"]
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment=os.environ["DEPLOYMENT_EMBEDDING"],
        azure_endpoint=saved_base,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    if saved_base:
        os.environ["AZURE_API_BASE"] = saved_base
        os.environ["AZURE_OPENAI_ENDPOINT"] = saved_base
    return AzureOpenAIEmbeddings(model=settings.emb_model_name)


def load_document(path: Path):
    """Load a document from the knowledge base based on file extension.
    
    Currently supports PDF files using PDFMinerLoader. The function can be
    extended to support additional formats like text and markdown files.
    
    Args:
        path (Path): Path to the document file to load.
        
    Returns:
        List[Document]: List of LangChain Document objects containing the
            loaded content and metadata.
            
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file extension is not supported.
        
    Note:
        Currently only PDF files are supported. Text and markdown support
        is commented out but can be enabled if needed.
    """

    if not path.exists():
        raise FileNotFoundError(f"Document {path} not found.")
    
    ext = path.suffix.lower()
  
    if ext == ".pdf":
        loader = PDFMinerLoader(str(path))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    """
    elif ext == ".txt":
        loader = TextLoader(str(path))
    elif ext in [".md", ".markdown"]:
        loader = UnstructuredMarkdownLoader(str(path))
    """
    return loader.load()


def split_documents(docs: List[Document], settings: RAG_Settings) -> List[Document]:
    """Split documents into smaller chunks for vector indexing.
    
    Uses RecursiveCharacterTextSplitter to break documents into manageable chunks
    with configurable size and overlap. The splitter uses a hierarchy of separators
    to maintain semantic coherence where possible.
    
    Args:
        docs (List[Document]): List of documents to split.
        settings (RAG_Settings): Configuration containing chunk_size and chunk_overlap.
        
    Returns:
        List[Document]: List of document chunks ready for embedding and indexing.
        
    Note:
        The splitter tries to split on paragraph breaks first, then sentences,
        and finally on smaller units to maintain context while respecting size limits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)


def recreate_collection_for_rag(client: QdrantClient, settings: RAG_Settings, vector_size: int):
    """Create or recreate a Qdrant collection optimized for RAG.
        
        Sets up a new Qdrant collection with optimized configuration for vector similarity
        search and text search. Includes HNSW indexing, quantization, and payload indices
        for efficient retrieval.
        
        Args:
            client (QdrantClient): Qdrant client instance.
            settings (RAG_Settings): Configuration containing collection name.
            vector_size (int): Dimensionality of the embedding vectors.
            
        Note:
            If a collection with the same name exists, it will be deleted and recreated.
            The collection is configured with COSINE distance, HNSW indexing for fast
            similarity search, and scalar quantization for memory efficiency.
        """
    
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)

    client.create_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=32, ef_construct=256),
        optimizers_config=OptimizersConfigDiff(default_segment_number=2),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=False)
        ),
    )

    client.create_payload_index(settings.collection, "text", PayloadSchemaType.TEXT)
    for key in ["doc_id", "source", "title", "lang"]:
        client.create_payload_index(settings.collection, key, PayloadSchemaType.KEYWORD)


def upsert_chunks(client: QdrantClient, settings: RAG_Settings, chunks: List[Document], embeddings: AzureOpenAIEmbeddings):
    """Embed and upsert document chunks into Qdrant in batches.
    
    Processes document chunks by generating embeddings and uploading them to Qdrant
    in configurable batch sizes to avoid payload size errors and manage API rate limits.
    
    Args:
        client (QdrantClient): Qdrant client instance.
        settings (RAG_Settings): Configuration containing batch size and collection name.
        chunks (List[Document]): List of document chunks to embed and store.
        embeddings (AzureOpenAIEmbeddings): Embeddings instance for vectorization.
        
    Note:
        Uses batched embedding to respect API rate limits and batched upserts
        to handle large document collections efficiently.
    """
    batch_size = settings.upsert_batch_size
    vecs = batched_embed_documents(chunks, embeddings, batch_size=batch_size, delay=1.0)
    points = build_points(chunks, vecs)

    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=settings.collection,
            points=batch,
            wait=True
        )
        print(f"⬆️  Upserted {i+len(batch)} / {total} points into Qdrant")
    print(f"✅ Finished upserting {total} points into Qdrant\n")


def batched_embed_documents(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, batch_size: int = 5, delay: float = 1.0) -> List[List[float]]:
    """Embed documents in small batches to avoid rate limits and API errors.
    
    Processes document chunks in batches to generate embeddings while respecting
    API rate limits. Includes retry logic and configurable delays between batches.
    
    Args:
        chunks (List[Document]): List of document chunks to embed.
        embeddings (AzureOpenAIEmbeddings): Embeddings instance for vectorization.
        batch_size (int, optional): Number of documents to process per batch. 
            Defaults to 5.
        delay (float, optional): Delay in seconds between batches. Defaults to 1.0.
        
    Returns:
        List[List[float]]: List of embedding vectors corresponding to each chunk.
        
    Note:
        Implements retry logic with exponential backoff for handling rate limits
        and temporary API failures.
    """
    all_vecs: List[List[float]] = []
    total = len(chunks)
    num_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        texts = [c.page_content for c in batch]

        success = False
        while not success:
            try:
                vecs = embeddings.embed_documents(texts)
                all_vecs.extend(vecs)
                success = True
                print(f"[Batch {batch_idx+1}/{num_batches}] Embedded chunks {start+1}–{end} of {total}")
            except Exception as e:
                print(f"[Batch {batch_idx+1}/{num_batches}] Rate limit or error: {e}")
                print("Waiting 10s before retry...")
                time.sleep(10)  # wait before retrying
        time.sleep(delay)  # throttle a bit between batches

    print(f"✅ Finished embedding {total} chunks in {num_batches} batches\n")
    return all_vecs


def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    """Build Qdrant point structures from document chunks and embeddings.
        
        Combines document chunks with their corresponding embedding vectors to create
        PointStruct objects suitable for insertion into Qdrant. Extracts metadata
        and assigns sequential IDs.
        
        Args:
            chunks (List[Document]): List of document chunks containing text and metadata.
            embeds (List[List[float]]): List of embedding vectors corresponding to chunks.
            
        Returns:
            List[PointStruct]: List of Qdrant points ready for insertion, each containing
                an ID, vector, and payload with document metadata and text content.
                
        Note:
            Point IDs start from 1 and increment sequentially. The payload includes
            document metadata plus chunk-specific information like chunk_id.
        """
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "doc_id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "lang": doc.metadata.get("lang", "en"),
            "text": doc.page_content,
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts



#################################################################### 
# Qdrant search functions
####################################################################


def hybrid_search(client: QdrantClient, settings: RAG_Settings, query: str, embeddings: AzureOpenAIEmbeddings):
    """Perform hybrid search combining semantic and text-based retrieval.
        
        Implements a sophisticated hybrid search strategy that combines:
        1. Semantic similarity search using vector embeddings
        2. Text-based keyword search using Qdrant's text matching
        3. Score fusion with configurable weights
        4. Optional Maximum Marginal Relevance (MMR) for result diversification
        
        Args:
            client (QdrantClient): Qdrant client instance.
            settings (RAG_Settings): Configuration containing search parameters.
            query (str): Search query string.
            embeddings (AzureOpenAIEmbeddings): Embeddings instance for query vectorization.
            
        Returns:
            List[ScoredPoint]: List of search results ranked by hybrid relevance score,
                limited to settings.final_k results.
                
        Note:
            The hybrid approach first retrieves semantic candidates, then applies text
            boost to results that also match keyword search. MMR diversification is
            applied if enabled to reduce redundancy in results.
        """
    sem = qdrant_semantic_search(client, settings, query, embeddings, limit=settings.top_n_semantic, with_vectors=True)
    if not sem: return []
    text_ids = set(qdrant_text_prefilter_ids(client, settings, query, settings.top_n_text))
    scores = [p.score for p in sem]
    smin, smax = min(scores), max(scores)
    def norm(x): return 1.0 if smax == smin else (x - smin) / (smax - smin)
    fused: List[Tuple[int, float, Any]] = []
    for idx, p in enumerate(sem):
        base = norm(p.score)
        fuse = settings.alpha * base
        if p.id in text_ids:
            fuse += settings.text_boost
        fused.append((idx, fuse, p))
    fused.sort(key=lambda t: t[1], reverse=True)
    if settings.use_mmr:
        qv = embeddings.embed_query(query)
        N = min(len(fused), max(settings.final_k * 5, settings.final_k))
        cut = fused[:N]
        vecs = [cast(List[float], sem[i].vector) for i, _, _ in cut]
        mmr_idx = mmr_select(qv, vecs, settings.final_k, settings.mmr_lambda)
        return [cut[i][2] for i in mmr_idx]
    return [p for _, _, p in fused[:settings.final_k]]


def qdrant_semantic_search(client: QdrantClient, settings: RAG_Settings, query: str, embeddings: AzureOpenAIEmbeddings, limit: int, with_vectors: bool = False):
    """Perform semantic similarity search in Qdrant using vector embeddings.
        
        Converts the query to a vector embedding and searches for similar vectors
        in the Qdrant collection using cosine similarity.
        
        Args:
            client (QdrantClient): Qdrant client instance.
            settings (RAG_Settings): Configuration containing collection name.
            query (str): Search query to vectorize and search for.
            embeddings (AzureOpenAIEmbeddings): Embeddings instance for query vectorization.
            limit (int): Maximum number of results to return.
            with_vectors (bool, optional): Whether to include vectors in results. 
                Defaults to False.
                
        Returns:
            List[ScoredPoint]: List of semantically similar documents with similarity scores.
            
        Note:
            Uses HNSW approximate search with configurable parameters for performance.
            Results are ordered by cosine similarity score in descending order.
        """
    qv = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=settings.collection,
        query=qv,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=SearchParams(hnsw_ef=256, exact=False),
    )
    return res.points


def qdrant_text_prefilter_ids(client: QdrantClient, settings: RAG_Settings, query: str, max_hits: int) -> List[int]:
    """Get document IDs that match text-based keyword search.
        
        Performs text matching search in Qdrant to find documents containing
        query keywords. Used as part of hybrid search to boost semantically
        similar results that also contain relevant keywords.
        
        Args:
            client (QdrantClient): Qdrant client instance.
            settings (RAG_Settings): Configuration containing collection name.
            query (str): Query string for text matching.
            max_hits (int): Maximum number of matching IDs to return.
            
        Returns:
            List[int]: List of document IDs that match the text query.
            
        Note:
            Uses Qdrant's scroll API with text filters to efficiently retrieve
            matching document IDs without loading full payloads or vectors.
        """
    matched_ids: List[int] = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(must=[FieldCondition(key="text", match=MatchText(text=query))]),
            limit=min(256, max_hits - len(matched_ids)),
            offset=next_page,
            with_payload=False,
            with_vectors=False,
        )
        matched_ids.extend([int(p.id) for p in points])
        if not next_page or len(matched_ids) >= max_hits:
            break
    return matched_ids


def mmr_select(query_vec: List[float], candidates_vecs: List[List[float]], k: int, lambda_mult: float) -> List[int]:
    """Select diverse results using Maximum Marginal Relevance (MMR).
        
        Implements MMR algorithm to balance relevance and diversity in search results.
        Iteratively selects documents that are relevant to the query while being
        dissimilar to already selected documents.
        
        Args:
            query_vec (List[float]): Query vector for relevance calculation.
            candidates_vecs (List[List[float]]): List of candidate document vectors.
            k (int): Number of documents to select.
            lambda_mult (float): Balance parameter between relevance (1.0) and 
                diversity (0.0). Typical values: 0.5-0.7.
                
        Returns:
            List[int]: Indices of selected documents in order of selection.
            
        Note:
            MMR score = λ * relevance - (1-λ) * max_similarity_to_selected
            Higher lambda_mult favors relevance over diversity.
        """
    import numpy as np
    V = np.array(candidates_vecs, dtype=float)
    q = np.array(query_vec, dtype=float)
    def cos(a, b):
        na = (a @ a) ** 0.5 + 1e-12
        nb = (b @ b) ** 0.5 + 1e-12
        return float((a @ b) / (na * nb))
    sims = [cos(v, q) for v in V]
    selected: List[int] = []
    remaining = set(range(len(V)))
    while len(selected) < min(k, len(V)):
        if not selected:
            best = max(remaining, key=lambda i: sims[i])
            selected.append(best)
            remaining.remove(best)
            continue
        best_idx = None
        best_score = -1e9
        for i in remaining:
            max_div = max([cos(V[i], V[j]) for j in selected]) if selected else 0.0
            score = lambda_mult * sims[i] - (1 - lambda_mult) * max_div
            if score > best_score:
                best_score = score
                best_idx = i
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def format_docs_for_prompt(points: Iterable[Any]) -> str:
    """Format document points for inclusion in LLM prompts with source citations.
        
        Converts Qdrant search result points into a formatted string suitable for
        use as context in RAG prompts. Each document includes source attribution
        for proper citation in generated responses.
        
        Args:
            points (Iterable[Any]): Iterable of Qdrant points containing document 
                payload with text and metadata.
                
        Returns:
            str: Formatted string with each document prefixed by its source,
                separated by double newlines for clear delineation.
                
        Note:
            Format: "[source:<source_name>] <document_text>" for each document.
            Unknown sources are labeled as "unknown".
        """
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        blocks.append(f"[source:{src}] {pay.get('text','')}")
    return "\n\n".join(blocks)
