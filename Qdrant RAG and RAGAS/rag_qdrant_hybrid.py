"""
RAG pipeline with Qdrant and AzureOpenAI embeddings

docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant 

"""

from __future__ import annotations
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from utils.utils import Settings, hybrid_search, build_rag_chain, get_llm, get_embeddings, format_docs_for_prompt, get_qdrant_client

from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PDFMinerLoader

from qdrant_client.models import ScalarType
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    PointStruct,
)

# Load env vars
load_dotenv()

settings = Settings()


# ========== Data prep ============================================================

def simulate_corpus() -> List[Document]:
    """Return a small fake corpus"""
    docs = [
        Document(
            page_content="LangChain is a framework for building LLM apps.",
            metadata={"id": "doc1", "source": "intro-langchain.md", "title": "Intro LangChain", "lang": "en"}
        ),
        Document(
            page_content="FAISS is a library for similarity search of dense vectors.",
            metadata={"id": "doc2", "source": "faiss-overview.md", "title": "FAISS Overview", "lang": "en"}
        ),
        Document(
            page_content="Sentence-transformers like MiniLM produce embeddings.",
            metadata={"id": "doc3", "source": "embeddings-minilm.md", "title": "MiniLM Embeddings", "lang": "en"}
        ),
        Document(
            page_content="A RAG pipeline includes indexing, retrieval and generation.",
            metadata={"id": "doc4", "source": "rag-pipeline.md", "title": "RAG Pipeline", "lang": "en"}
        ),
        Document(
            page_content="MMR balances relevance and diversity in retrieval.",
            metadata={"id": "doc5", "source": "retrieval-mmr.md", "title": "MMR Retrieval", "lang": "en"}
        ),
    ]
    return docs

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Split docs into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)


# ========== Qdrant Collections ============================================================

def recreate_collection_for_rag(client: QdrantClient, settings: Settings, vector_size: int):
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
                print("Waiting 60s before retry...")
                time.sleep(60)  # wait before retrying
        time.sleep(delay)  # throttle a bit between batches

    print(f"✅ Finished embedding {total} chunks in {num_batches} batches")
    return all_vecs

def upsert_chunks(client: QdrantClient, settings: Settings, chunks: List[Document], embeddings: AzureOpenAIEmbeddings, batch_size: int = 50):
    """Embed and upsert chunks into Qdrant in batches to avoid payload size errors"""
    vecs = batched_embed_documents(chunks, embeddings, batch_size=5, delay=1.0)
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


# ========== Main ============================================================

def load_document(doc_name: str):
    """Load a document from knowledge base depending on its extension."""

    path = Path(f"knowledge_base/{doc_name}")  # adjust to your KB path
    if not path.exists():
        raise FileNotFoundError(f"Document {doc_name} not found in knowledge base.")
    
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        loader = PDFMinerLoader(str(path))
    elif ext == ".txt":
        loader = TextLoader(str(path))
    elif ext in [".md", ".markdown"]:
        loader = UnstructuredMarkdownLoader(str(path))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return loader.load()


def main():
    """Demo full RAG pipeline"""
    s = settings
    embeddings = get_embeddings(s)
    llm = get_llm(s)
    client = get_qdrant_client(s)

    #### Load from Knowledge Base #######
    doc_name = "Short AI Act"
    doc_name = "EU AI Act.pdf"
    if doc_name == "":
        docs = simulate_corpus()
        questions = [
                    "Cos'è una pipeline RAG e quali sono le sue fasi?",
                    "A cosa serve FAISS e che caratteristiche offre?",
                    "Che cos'è MMR e perché riduce la ridondanza?",
                    "Qual è la dimensione degli embedding di all-MiniLM-L6-v2?",
                    ]
    else:
        docs = load_document(doc_name)
        print(f"Loaded {len(docs)} documents from {doc_name}.")
        questions = [
                    "Does the EU AI Act apply to AI systems developed outside the EU but used within the EU?",
                    "What are the obligations for providers of high-risk AI systems?",
                    "Who enforces the EU AI Act?",
                    "When will the EU AI Act enter into force?",
                    ]
    #####################################

    s.collection = doc_name
    chunks = split_documents(docs, s)
    print(f"Chunked document into {len(chunks)} chunks:")
    for i in range(5):
        print(f"\n {chunks[i]}")
    vector_size = len(embeddings.embed_query("hello world"))
    recreate_collection_for_rag(client, s, vector_size)
    upsert_chunks(client, s, chunks, embeddings)

    for q in questions:
        hits = hybrid_search(client, s, q, embeddings)
        print("=" * 80)
        print("Q:", q)
        if not hits:
            print("Nessun risultato.")
            continue
        for p in hits:
            print(f"- id={p.id} score={p.score:.4f} src={p.payload.get('source')}")
        if llm:
            try:
                ctx = format_docs_for_prompt(hits)
                chain = build_rag_chain(llm)
                answer = chain.invoke({"question": q, "context": ctx})
                print("\n", answer, "\n")
            except Exception as e:
                print(f"\nLLM failed: {e}")
                print("\nContenuto recuperato:\n")
                print(format_docs_for_prompt(hits))
                print()
        else:
            print("\nContenuto recuperato:\n")
            print(format_docs_for_prompt(hits))
            print()

if __name__ == "__main__":
    main()

"""
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant 
"""