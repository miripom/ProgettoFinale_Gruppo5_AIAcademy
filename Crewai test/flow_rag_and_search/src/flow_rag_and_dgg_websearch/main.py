import os
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, router, listen
from flow_rag_and_dgg_websearch.crews.rag_crew.rag_crew import RagCrew
from flow_rag_and_dgg_websearch.crews.search_crew.search_crew import SearchCrew
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant, FAISS
from langchain_openai import AzureOpenAIEmbeddings
import pandas as pd  # <-- necessario per mlflow.evaluate con dataframe

import mlflow
from mlflow import metrics
from mlflow.metrics import genai
import mlflow.tracing

import time
from datetime import datetime

import evaluate

# First, run the MLflow server with:
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
# OR run it in background with:
# Start-Process mlflow -ArgumentList "server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"

# Then setup the dashboard
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

mlflow.autolog()  # autolog per openai/langchain/crewai dove supportato
mlflow.tracing.enable()
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("rag_or_search")

load_dotenv()
faiss_0__or__qdrant_1 = int(os.environ["FAISS_0__OR__QDRANT_1"])


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


if faiss_0__or__qdrant_1 == 0:

    path_to_indices_folder = "C:/Users/LH668YN/OneDrive - EY/Desktop/CrewAI/3_Flow_Rag_and_DDG_WebSearch/Local_RAG/indices"
    rag_index = FAISS.load_local(path_to_indices_folder, embeddings, allow_dangerous_deserialization=True)


elif faiss_0__or__qdrant_1 == 1:

    client = QdrantClient(host="localhost", port=6333)
    rag_index = Qdrant(
        client=client,
        collection_name="index_risposte-sbagliate",
        embeddings=embeddings,
    )


class FindingsState(BaseModel):
    """State container for the WebSearch flow."""
    user_query: str = ""
    summary: dict = {"rag": "", "web": ""}
    answer: str = ""
    context: str = ""          # se vuoi passare contesto ai judge
    ground_truth: str = ""     # opzionale: se vuoi attivare similarity/correctness


class RagVsSearchFlow(Flow[FindingsState]):

    @start()
    def ask_query(self):
        query = input("Fai la tua richiesta: ")
        if not query:
            query = "A che numero sta pensando Luca?"
            self.state.ground_truth = "Luca sta pensando al numero 42."
            print(f"➡️ Question di esempio: {query}")
        self.state.user_query = query

        # Log query di input
        mlflow.log_param("user_query", query)
        mlflow.log_metric("user_query_length_chars", len(query))
        mlflow.log_metric("user_query_length_words", len(query.split()))

        return query

    @router(ask_query)
    def route_query(self, query: str):
        print(f"The query is {query}")
        docs = rag_index.similarity_search_with_score(query, k=1)
        best_doc, score = docs[0]
        print(f"Best similarity score: {score:.3f}")
        print(f"With doc: {best_doc}")

        if score < 0.70:
            return "search"
        elif score < 0.85:
            return "both"
        else:
            return "rag"


    @listen("search")
    def run_search(self, query: str):

        started = time.perf_counter()

        print("RAG non adatta a rispondere a questo prompt. Procedo con ricerca web.")
        search_crew = SearchCrew().crew()
        result = search_crew.kickoff(inputs={"query": query})

        duration = time.perf_counter() - started

        self.state.summary["web"] = str(result)

        # Metriche/artefatti ricerca
        mlflow.log_metric("search_duration_seconds", duration)
        mlflow.log_metric("search_results_chars", len(self.state.summary["web"]))
        mlflow.log_metric("search_results_words", len(self.state.summary["web"].split()))
        mlflow.log_metric("search_results_lines", self.state.summary["web"].count("\n") + 1 if self.state.summary["web"] else 0)
        mlflow.log_text(self.state.summary["web"], "search_summary.txt")

        print("\n--- Risultato Web ---")
        print(self.state.summary["web"])
        return self.state.summary["web"]
    
    @listen("both")
    def run_both(self, query: str):
        print("RAG solo parzialmente adatta a rispondere a questo prompt. Procedo anche con ricerca web.")
        search_crew = SearchCrew().crew()
        rag_crew = RagCrew().crew()
        search_summary = search_crew.kickoff(inputs={"query": query})
        rag_summary = rag_crew.kickoff(inputs={"query": query})
        print("\n--- Risultato Combinato RAG + WebSearch ---")
        print(f"RAG: {rag_summary}\n\nWEB: {search_summary}")

        context_parts = []
        if self.state.summary.get("rag"):
            context_parts.append(f"RAG Context:\n{self.state.summary['rag']}")
        if self.state.summary.get("web"):
            context_parts.append(f"WEB Context:\n{self.state.summary['web']}")
        joined_summary = "\n\n".join(context_parts) if context_parts else ""
        return joined_summary

    @listen("rag")
    def run_rag(self, query: str):

        started = time.perf_counter()

        print("RAG adatta a rispondere a questo prompt.")
        rag_crew = RagCrew().crew()
        result = rag_crew.kickoff(inputs={"query": query})

        duration = time.perf_counter() - started
        self.state.summary["rag"] = str(result)
        
        # Metriche/artefatti ricerca
        mlflow.log_metric("rag_duration_seconds", duration)
        mlflow.log_metric("rag_results_chars", len(self.state.summary["rag"]))
        mlflow.log_metric("rag_results_words", len(self.state.summary["rag"].split()))
        mlflow.log_metric("rag_results_lines", self.state.summary["rag"].count("\n") + 1 if self.state.summary["rag"] else 0)
        mlflow.log_text(self.state.summary["rag"], "rag_summary.txt")

        print("\n--- Risultato RAG ---")
        print(self.state.summary["rag"])
        return self.state.summary["rag"]
    

    @listen("run_rag")
    @listen("run_both")
    @listen("run_search")
    def display_results(self, summary: str):
        print("\n📋 RISULTATI\n")
        print(f"🔍 Query: {self.state.user_query}\n")
        print(summary)

        # Build context depending on source

        # --- LLM-as-a-judge con MLflow ---
        try:
            os.environ["OPENAI_API_BASE"] = "https://aiacademymainlucamaci.openai.azure.com/"
            os.environ["openai_api_base"] = "https://aiacademymainlucamaci.openai.azure.com/"

            os.environ["OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"
            os.environ["openai_deployment_name"] = "gpt-4o"

            os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
            os.environ["openai_api_version"] = "2024-12-01-preview"
            
            eval_metrics = self._run_llm_judge_mlflow(
                user_query=self.state.user_query,
                prediction=summary,
                context=self.state.context or None,              # opzionale
                ground_truth=self.state.ground_truth or None,    # opzionale
            )
            # Salvo snapshot metriche anche come dict (facile da leggere)
            if eval_metrics:
                mlflow.log_dict(eval_metrics, "eval_metrics_snapshot.json")
                mlflow.set_tag("llm_judge_status", "success")
        except Exception as e:
            mlflow.set_tag("llm_judge_status", f"failed:{type(e).__name__}")
            mlflow.log_text(str(e), "llm_judge_error.txt")

        return "Flow completato!"
    

        # ---------- Nuovo: judge con mlflow.evaluate ----------
    def _run_llm_judge_mlflow(
        self,
        user_query: str,
        prediction: str,
        context: str | None = None,
        ground_truth: str | None = None,
    ):
        """
        Usa i judge integrati MLflow:
          - answer_relevance (richiede inputs+predictions)
          - faithfulness (se fornisci context)
          - answer_similarity/answer_correctness (se fornisci ground_truth)
          - toxicity (metric non-LLM)
        Le metriche e la tabella vengono loggate automaticamente nel run attivo.
        """
        # Tabella di valutazione a 1 riga (scalabile a molte righe)
        data = {
            "inputs": [user_query],
            "predictions": [prediction],
        }
        if context is not None:
            data["context"] = [context]
        if ground_truth is not None:
            data["ground_truth"] = [ground_truth]

        df = pd.DataFrame(data)

        # Costruisci lista metriche in base alle colonne disponibili
        extra_metrics = [
            genai.answer_relevance(),  # sempre se hai inputs+predictions
            metrics.toxicity(),                # metrica non-LLM (HF pipeline)
        ]
        if "context" in df.columns:
            extra_metrics.append(genai.faithfulness())
        if "ground_truth" in df.columns:
            extra_metrics.extend([
                genai.answer_similarity(),
                genai.answer_correctness(),
            ])

        # model_type:
        # - "text" va bene per generico testo
        # - "question-answering" se passi ground_truth in stile QA
        model_type = "question-answering" if "ground_truth" in df.columns else "text"

        results = mlflow.evaluate(
            data=df,
            predictions="predictions",
            targets="ground_truth" if "ground_truth" in df.columns else None,
            model_type=model_type,
            extra_metrics=extra_metrics,
            evaluators="default",
        )
        # MLflow ha già loggato metriche e tabella 'eval_results_table'
        return results.metrics

def kickoff():
    flow = RagVsSearchFlow()
    flow.plot("tool_choice_flow")
    flow.kickoff()

if __name__ == "__main__":
    kickoff()
