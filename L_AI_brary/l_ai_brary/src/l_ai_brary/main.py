#!/usr/bin/env python
import os
import time
import mlflow
import pandas as pd
from crewai import LLM
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any, Literal
from crewai.flow.flow import Flow, start, router, listen, or_
from l_ai_brary.crews.image_crew.image_crew import ImageCrew
from l_ai_brary.crews.sanitize_crew.sanitize_crew import SanitizeCrew
from l_ai_brary.crews.rag_and_search_crew.rag_and_search_crew import RagAndSearchCrew


load_dotenv()  # take environment variables from .env.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.autolog()  # autolog per openai/langchain/crewai dove supportato
mlflow.set_experiment("L_AI_brary")

class ChatState(BaseModel):
    chat_history: list[dict] = []
    user_input: str = ""
    assistant_response: list = [] # might contain text and images. in streamlit we can iterate over it to display them in the order they appear in here
    messages_count: int = 0
    needs_refresh: bool = False  # Flag to indicate UI needs refresh
    user_quit: bool = False
    sanitized_result: str = ""
    summary: str = ""
    counts: int = 0
    context: str = ""          
    ground_truth: str = "" 


class ChatbotFlow(Flow[ChatState]):

    @start("new_turn")
    def wait_for_input(self):
        # Cleanup from previous turn

        print(" Inside wait_for_input")
        print(f"with query: {self.state.user_input}")

        self.state.user_input = ""
        self.state.assistant_response = []

        while self.state.user_input == "":
            time.sleep(.1)  # wait for user input to be set in Streamlit
            if self.state.user_quit:
                print(" User requested to quit. Exiting flow.")
                return None
        
        if self.state.user_input:
            print(f" added input to chat history: {self.state.user_input}")
            # Append user message
            self.append_user_message(self.state.user_input)
            self.state.counts += 1
        mlflow.log_param(f"user_query_{self.state.counts}", self.state.user_input)
        mlflow.log_metric("user_query_length_chars", len(self.state.user_input))
        mlflow.log_metric("user_query_length_words", len(self.state.user_input.split()))
        return self.state


    @router(wait_for_input)
    def route_user_input(self):
        """Decide what to do with the user query."""
        print(" Inside route_user_input")
        print(f"with query: {self.state.user_input}")
        query = self.state.user_input

        self.state.sanitized_result = SanitizeCrew().crew().kickoff(inputs={"user_input": query})
        print(" SanitizeCrew result: ", self.state.sanitized_result)

        llm = LLM(model='azure/gpt-4o')
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a binary classifier that routes user queries (after they have been sanitized) to the appropriate service.\n"
                    f"Analyze the sanitized query: '{self.state.sanitized_result}' and answer as follows:\n"
                    f"1 - If the sanitized query is asking to create, generate, draw, or in general it is asking for a visual content about literature (comics included), return 'image';\n"
                    f"2 - If the sanitized query is asking for information, summaries, facts, or discussion about books, authors, plots, genres, comics or the literature domain in general (do not include generation, drawing, and visualization of images of scene or characters in this step), return 'rag_and_search';\n"
                    f"3 - If none of the above, return 'new_turn'.\n"
                    f"Only return the labels 'image', 'rag_and_search', or 'new_turn' and NEVER say anything else.\n"
                    f"Do NOT consider specific copyrighted names as preventing an 'image' classification. Focus on the user's intent to generate a visual."
                )
            }
        ]

 
        classification = llm.call(messages=messages)
        print('CLASSIFICATION: ', classification)

        if not query:
            time.sleep(2)
            return "new_turn"  # loop back until user says something
        
        # Simple routing logic — later can be replaced by an LLM-based router
        if classification.lower() == "image":
            self.append_agent_response(self.state.sanitized_result.raw)
            return "image"
        elif classification.lower() == "rag_and_search":
            self.append_agent_response(self.state.sanitized_result)
            return "rag_and_search"
        else:
            self.append_agent_response(self.state.sanitized_result.raw)
            return "new_turn"




    @listen("rag_and_search")
    @router(route_user_input)
    def do_rag_and_search(self):
        # call RAG crew
        # started = time.perf_counter()
        # crew = RagAndSearchCrew().crew()
        # answer = crew.kickoff(inputs={'input': self.state.sanitized_result})
        # print(" Inside do_rag_and_search")
        # self.append_agent_response(answer, "text")
        # duration = time.perf_counter() - started
        # self.state.summary = str(answer.raw)

        # Metriche/artefatti ricerca
        # mlflow.log_metric("search_duration_seconds", duration)
        # mlflow.log_metric("search_results_chars", len(self.state.summary))
        # mlflow.log_metric("search_results_words", len(self.state.summary.split()))
        # mlflow.log_metric("search_results_lines", self.state.summary.count("\n") + 1 if self.state.summary else 0)
        # mlflow.log_text(self.state.summary, "search_summary.txt")        
        return self.state


    @listen("image")
    @router(route_user_input)
    def generate_image(self):
        started = time.perf_counter()
        crew = ImageCrew().crew()
        path = crew.kickoff(inputs={'topic': self.state.sanitized_result.raw})

        self.append_agent_response(path.raw, "image")
        duration = time.perf_counter() - started
        self.state.summary = str(path.raw)

        # Metriche/artefatti ricerca
        mlflow.log_metric("search_duration_seconds", duration)
        mlflow.log_metric("search_results_chars", len(self.state.summary))
        mlflow.log_metric("search_results_words", len(self.state.summary.split()))
        mlflow.log_metric("search_results_lines", self.state.summary.count("\n") + 1 if self.state.summary else 0)
        mlflow.log_text(self.state.summary, "search_summary.txt")        
        return self.state

    @listen(or_(do_rag_and_search, generate_image))
    @router(route_user_input)
    def display_results(self):
        print("\n📋 RISULTATI DELLA RICERCA WEB\n")
        print(f"🔍 Query: {self.state.user_input}\n")
        print(self.state.summary)

        # --- LLM-as-a-judge con MLflow ---
        try:
            os.environ["OPENAI_API_BASE"] = "https://aiacademymainlucamaci.openai.azure.com/"
            os.environ["openai_api_base"] = "https://aiacademymainlucamaci.openai.azure.com/"

            os.environ["OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"
            os.environ["openai_deployment_name"] = "gpt-4o"

            os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
            os.environ["openai_api_version"] = "2024-12-01-preview"

            eval_metrics = self._run_llm_judge_mlflow(
                user_query=self.state.user_input,
                prediction=self.state.summary,
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

        return "new_turn"

    # ---------- judge con mlflow.evaluate ----------
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
            mlflow.metrics.genai.answer_relevance(),  # sempre se hai inputs+predictions
            mlflow.metrics.toxicity(),                # metrica non-LLM (HF pipeline)
        ]
        if "context" in df.columns:
            extra_metrics.append(mlflow.metrics.genai.faithfulness(context_column="context"))
        if "ground_truth" in df.columns:
            extra_metrics.extend([
                mlflow.metrics.genai.answer_similarity(),
                mlflow.metrics.genai.answer_correctness(),
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

    def append_agent_response(self, response: str | Any, type: Literal["text", "image", "md_file"] = "text"):
        self.state.assistant_response.append(response)
        self.state.chat_history.append({"role": "assistant", "content": response, "type": type})
        self.state.messages_count += 1
        self.state.needs_refresh = True

    def append_user_message(self, message: str):
        self.state.chat_history.append({"role": "user", "content": message})
        self.state.messages_count += 1
        self.state.needs_refresh = True

def kickoff():
    run_started_wall_time = datetime.utcnow().isoformat() + "Z"
    run_timer = time.perf_counter()
    with mlflow.start_run(nested=True):
        mlflow.set_tag("app_name", "guide_creator_flow")
        mlflow.set_tag("flow_name", "WebSearchFlow")
        mlflow.set_tag("run_started_at_utc", run_started_wall_time)
        mlflow.set_tag("environment", os.getenv("APP_ENV", "local"))

        chatbot_flow = ChatbotFlow()
        chatbot_flow.kickoff()
        mlflow.log_metric("run_duration_seconds", time.perf_counter() - run_timer)


def plot():
    chatbot_flow = ChatbotFlow()
    chatbot_flow.plot()


if __name__ == "__main__":
    kickoff()
