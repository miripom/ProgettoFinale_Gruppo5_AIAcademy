import os
import getpass
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader


# Avvia prima Qdrant con Docker:
# docker run -p 6333:6333 qdrant/qdrant

client = QdrantClient(host="localhost", port=6333)


CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY_PATH = os.path.dirname(CURRENT_FILE_PATH)
DOTENV_PATH = "C:/Users/LH668YN/OneDrive - EY/Desktop/CrewAI/3_Flow_Rag_and_DDG_WebSearch/flow_rag_and_dgg_websearch/.env"
load_dotenv(DOTENV_PATH)

faiss_0__or__qdrant_1 = int(os.environ["FAISS_0__OR__QDRANT_1"])

azure_openai_key = os.getenv("AZURE_API_KEY") or ""
azure_openai_endpoint = os.getenv("AZURE_API_BASE") or ""
api_version = os.getenv("AZURE_API_VERSION") or ""

deployment_embedding = os.getenv("DEPLOYMENT_EMBEDDING") or ""


# Modello embedding
embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")

loader = TextLoader(f"{CURRENT_DIRECTORY_PATH}/knowledge_base/rispostesbagliate.md", encoding="utf-8")
docs = loader.load()

"""Hardcoded splitter"""
split_doc = []
for d in docs:
    parts = d.page_content.split("---")
    for part in parts:
        part = part.strip()
        if part:
            split_doc.append(Document(page_content=part, metadata=d.metadata))


print("After splitting, here are the individual chunks obtained from the Knowledge Base:\n")
for chunk in split_doc:
  print(chunk)
  print("--------------------------------------------------------------------------------------")


####################### SALVATAGGIO DEL VECTOR STORE ############################################


if faiss_0__or__qdrant_1 == 0:
    print("ciao")
    vector_store = FAISS.from_documents(
        documents=split_doc,
        embedding=embeddings
    )
    vector_store.save_local(f"{CURRENT_DIRECTORY_PATH}/indices/faiss_index_risposte-sbagliate")


elif faiss_0__or__qdrant_1 == 1:
    
    vector_store = Qdrant.from_documents(
        documents=split_doc,
        embedding=embeddings,
        location="http://localhost:6333",
        collection_name="index_risposte-sbagliate",
    )

print(f"Loaded {len(split_doc)} chunks into the vector_store.")

