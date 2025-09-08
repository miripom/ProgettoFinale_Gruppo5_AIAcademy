from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Tuple, cast
from dataclasses import dataclass

from l_ai_brary import main as m
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
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



@dataclass
class RAG_Settings:
    """Config settings for RAG pipeline"""
    qdrant_url: str = "http://localhost:6333"        # Qdrant URL
    collection: str = "rag_chunks"                   # Collection name
    upsert_batch_size: int = 100                     # Upsert batch size
    emb_model_name: str = "text-embedding-ada-002"   # Embedding model
    chunk_size: int = 1000                           # Chunk size
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
    """Return Qdrant client"""
    return QdrantClient(url=settings.qdrant_url, timeout=60)


def get_embeddings(settings: RAG_Settings) -> AzureOpenAIEmbeddings:
    """Return Azure OpenAI embeddings"""
    return AzureOpenAIEmbeddings(model=settings.emb_model_name)


def load_document(path: Path):
    """Load a document from knowledge base depending on its extension."""

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
    """Split docs into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)


def recreate_collection_for_rag(client: QdrantClient, settings: RAG_Settings, vector_size: int):
    """(Re)create Qdrant collection with indices"""
    
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
  """Embed and upsert chunks into Qdrant in batches to avoid payload size errors"""

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
    """Embed documents in small batches to avoid rate limits"""
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
    """Build Qdrant points"""
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
    """Hybrid search with semantic + text + MMR"""
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
    """Semantic search in Qdrant"""
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
    """Return ids matching text filter"""
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
    """Select diverse results with MMR"""
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
    """Format docs with sources"""
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        blocks.append(f"[source:{src}] {pay.get('text','')}")
    return "\n\n".join(blocks)
