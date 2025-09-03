from __future__ import annotations
import os
from typing import List, Any, Iterable, Tuple, cast

from dotenv import load_dotenv
from dataclasses import dataclass

from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    MatchText,
    Filter,
    SearchParams,
)

load_dotenv()


# ========== Settings ==========

@dataclass
class Settings:
    """Config settings for RAG pipeline"""
    qdrant_url: str = "http://localhost:6333"  # Qdrant URL
    collection: str = "rag_chunks"             # Collection name
    emb_model_name: str = "text-embedding-ada-002"  # Embedding model
    chunk_size: int = 700                      # Chunk size
    chunk_overlap: int = 120                   # Overlap size
    top_n_semantic: int = 30                   # Candidates for semantic search
    top_n_text: int = 100                      # Candidates for text search
    final_k: int = 6                           # Final results count
    alpha: float = 0.75                        # Semantic weight
    text_boost: float = 0.20                   # Text boost
    use_mmr: bool = True                       # Use MMR diversification
    mmr_lambda: float = 0.6                    # MMR balance
    lm_base_env: str = "OPENAI_BASE_URL"       # LLM base URL env
    lm_key_env: str = "OPENAI_API_KEY"         # LLM API key env
    lm_model_env: str = "LMSTUDIO_MODEL"       # LLM model env


# ========== Embeddings & LLM ==========

def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """Return Azure OpenAI embeddings"""
    return AzureOpenAIEmbeddings(model=settings.emb_model_name)

def get_llm(settings: Settings):
    """Initialize LLM if configured"""
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
    """Build RAG chain with citations"""
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


# ========== Setup Qdrant ==========

def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Return Qdrant client"""
    return QdrantClient(url=settings.qdrant_url, timeout=60)

def hybrid_search(client: QdrantClient, settings: Settings, query: str, embeddings: AzureOpenAIEmbeddings):
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

def qdrant_semantic_search(client: QdrantClient, settings: Settings, query: str, embeddings: AzureOpenAIEmbeddings, limit: int, with_vectors: bool = False):
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

def qdrant_text_prefilter_ids(client: QdrantClient, settings: Settings, query: str, max_hits: int) -> List[int]:
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
