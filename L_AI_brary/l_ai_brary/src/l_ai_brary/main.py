#!/usr/bin/env python
"""L_AI_brary Main Application Module.

This module implements the main chatbot flow application for L_AI        while self.state.user_input == "":
            time.sleep(0.5)  # wait for user input to be set in Streamlit
            if self.state.user_quit:
                print(" User requested to quit. Exiting flow.")
                return "quit_flow"
        
        if self.state.user_input:
            print(f" Processing user input: {self.state.user_input}")
            # ✅ DON'T add user message here - it's already added in Streamlit for immediate display
            # self.append_user_message(self.state.user_input)  # REMOVED to prevent duplication
            self.state.counts += 1iterary
assistant that combines RAG (Retrieval-Augmented Generation), image generation,
and content sanitization capabilities. The application uses CrewAI flows to orchestrate
different AI agents and tools based on user input classification.

The application provides:
- Text sanitization for safe user input processing
- RAG-based search and information retrieval from literary content
- AI-powered image generation for literary themes
- MLflow integration for experiment tracking and evaluation
- LLM-as-a-judge evaluation system

Dependencies:
    - CrewAI for agent orchestration and flows
    - MLflow for experiment tracking and model evaluation
    - Azure OpenAI for LLM and embedding services
    - Qdrant for vector database operations
"""

import os
import time
import mlflow
import pandas as pd
from typing import Any, ClassVar, Literal
from crewai import LLM
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, router, listen, or_
from l_ai_brary.crews.image_crew.image_crew import ImageCrew
from l_ai_brary.crews.sanitize_crew.sanitize_crew import SanitizeCrew
from l_ai_brary.crews.rag_and_search_crew.rag_and_search_crew import RagAndSearchCrew
import threading
 
load_dotenv()  # take environment variables from .env.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.autolog()  # autolog per openai/langchain/crewai dove supportato
mlflow.set_experiment("L_AI_brary")
 
 
class ChatState(BaseModel):
    """State management model for the chatbot flow conversation.
    
    This Pydantic model maintains the complete state of a chatbot conversation,
    including chat history, user inputs, assistant responses, and various metrics
    for tracking and evaluation purposes.
    
    Attributes:
        chat_history (list[dict]): Complete conversation history with role and content.
        user_input (str): Current user input being processed.
        assistant_response (list): List of assistant responses (text, images, files).
        messages_count (int): Total number of messages in the conversation.
        needs_refresh (bool): Flag indicating if UI needs to be refreshed.
        user_quit (bool): Flag indicating if user wants to quit the conversation.
        sanitized_result (str): Sanitized version of user input.
        summary (str): Summary of the latest assistant response.
        counts (int): Counter for processed queries.
        context (str): Additional context for evaluation purposes.
        ground_truth (str): Ground truth data for evaluation metrics.
        LLMclassifier (ClassVar[LLM]): Shared LLM instance for input classification.
        sanitizer_crew (ClassVar): Shared crew for input sanitization.
        image_crew (ClassVar): Shared crew for image generation.
        rag_and_search_crew (ClassVar): Shared crew for RAG and search operations.
    """
    
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
 
    # Mark these as ClassVar so Pydantic ignores them
    LLMclassifier: ClassVar[LLM] = LLM(model='azure/gpt-4o')
    sanitizer_crew: ClassVar = SanitizeCrew().crew()
    image_crew: ClassVar = ImageCrew().crew()
    rag_and_search_crew: ClassVar = RagAndSearchCrew().crew()
 
class ChatbotFlow(Flow[ChatState]):
    """Main chatbot flow orchestrating AI agents for literary assistance.
    
    This CrewAI Flow manages the complete conversation workflow for L_AI_brary,
    routing user inputs to appropriate specialized crews based on content classification.
    The flow handles input sanitization, classification, and routing to either
    RAG-based search, image generation, or requesting new input.
    
    The flow implements a state machine with the following phases:
    1. Wait for user input and sanitization
    2. Classify input to determine appropriate service
    3. Execute the selected service (RAG search or image generation)
    4. Display results and evaluate with LLM-as-a-judge
    5. Return to waiting for new input
    
    Attributes:
        state (ChatState): The conversation state containing all flow data.
    """
 
    @start("new_turn")
    def wait_for_input(self):
        """Wait for and process new user input to start a conversation turn.
        
        This method handles the initialization of a new conversation turn by:
        - Clearing previous turn data
        - Waiting for user input via polling
        - Adding user input to chat history
        - Logging user query metrics to MLflow
        
        Returns:
            str: "quit_flow" if user wants to quit, "continue_flow" otherwise.
            
        Note:
            This method polls for user input with a 0.5 second delay until input
            is provided or user requests to quit.
        """
        # Cleanup from previous turn
 
        print(" Inside wait_for_input")
        print(f"with query: {self.state.user_input}")
 
        self.state.user_input = ""
        self.state.assistant_response = []
 
        while self.state.user_input == "":
            time.sleep(0.5)  # wait for user input to be set in Streamlit
            if self.state.user_quit:
                print(" User requested to quit. Exiting flow.")
                return "quit_flow"
       
        if self.state.user_input:
            print(f" Processing user input: {self.state.user_input}")
            # ✅ DON'T add user message here - it's already added in Streamlit for immediate display
            # self.append_user_message(self.state.user_input)  # REMOVED to prevent duplication
            self.state.counts += 1
        mlflow.log_param(f"user_query_{self.state.counts}", self.state.user_input)
        mlflow.log_metric("user_query_length_chars", len(self.state.user_input))
        mlflow.log_metric("user_query_length_words", len(self.state.user_input.split()))
       
        return "continue_flow"
 
 
    @router(wait_for_input)
    def route_user_input(self, quit_or_continue: str):
        """Route user input to appropriate service based on LLM classification.
        
        This method performs the following workflow:
        1. Sanitizes user input using the sanitizer crew
        2. Classifies the sanitized input using an LLM classifier
        3. Routes to appropriate service based on classification
        
        The classifier determines if the query is requesting:
        - Image generation (literary domain images)
        - RAG search (information about books/literature)
        - New input (non-literary domain queries)
        
        Args:
            quit_or_continue (str): Flow control parameter from previous step.
            
        Returns:
            str: One of "quit_flow", "image", "rag_and_search", or "new_turn"
                based on the classification result.
                
        Note:
            The sanitization step helps ensure safe input processing before
            classification and prevents potential prompt injection attacks.
        """
        print(" Inside route_user_input")
        print(f"with query: {self.state.user_input}")
 
        if quit_or_continue == "quit_flow":
            return "quit_flow"
 
        query = self.state.user_input


        result = self.state.sanitizer_crew.kickoff(inputs={"user_input": query})
        self.state.sanitized_result = str(result.raw)
        print(" SanitizeCrew result: ", self.state.sanitized_result)

        messages = [
            {
                "role": "system",
                "content": (
 
                    f"You are a binary classifier that routes user queries (after they have been sanitized) to the appropriate service."
                    f"Analyze the sanitized query: '{self.state.sanitized_result}'; "
                    f"Analyze the sanitized query: '{self.state.sanitized_result}'; "
                    f"If the sanitized query is asking for the generation of an image pertaining to the literary domain, return 'image' (without the '); "
                    f"If the sanitized query is asking for information about a book or about the literature domain in general, return 'rag_and_search' (without the ');"
                    f"If the sanitized query is asking for a new input from the user, as it doesn't pertain to the literary domain, return 'new_turn' (without the ');"
                    f"Only return the labels 'image', 'rag_and_search', 'new_turn' (without the ' surrounding them) and NEVER say anything else."
                    f"Don't focus on copyright issues (they will be handled later), just focus on the content of the query."
 
                )
            }
        ]
 
 
        classification = self.state.LLMclassifier.call(messages=messages)
        print('CLASSIFICATION: ', classification)
       
        # Simple routing logic — later can be replaced by an LLM-based router
        if classification.lower() == "image":
            # self.append_agent_response(self.state.sanitized_result)
            return "image"
        elif classification.lower() == "rag_and_search":
            # self.append_agent_response(self.state.sanitized_result)
            return "rag_and_search"
        else:
            self.append_agent_response(self.state.sanitized_result)
            return "new_turn"


    @listen("rag_and_search")
    @router(route_user_input)
    def do_rag_and_search(self):
        """Execute RAG-based search and information retrieval.
        
        This method handles queries requesting information about books or literature
        by invoking the RAG and search crew. It performs semantic search over
        indexed literary content and generates comprehensive responses.
        
        The method:
        1. Measures execution time for performance tracking
        2. Executes the RAG and search crew with the user query
        3. Logs performance metrics to MLflow
        4. Appends the response to the conversation
        
        Returns:
            str: Always returns "new_turn" to continue the conversation flow.
            
        Note:
            All metrics including duration, response length, and content are
            automatically logged to MLflow for monitoring and evaluation.
        """
        # call RAG crew
        started = time.perf_counter()
 
        print(" Inside do_rag_and_search")
 
        result = self.state.rag_and_search_crew.kickoff(inputs={"query": self.state.user_input})
        duration = time.perf_counter() - started
        self.state.summary = str(result.raw)
        self.append_agent_response(self.state.summary, "text")

        mlflow.log_metric("rag_duration_seconds", duration)
        mlflow.log_metric("rag_results_chars", len(self.state.summary))
        mlflow.log_metric("rag_results_words", len(self.state.summary.split()))
        mlflow.log_metric("rag_results_lines", self.state.summary.count("\n") + 1 if self.state.summary else 0)
        mlflow.log_text(self.state.summary, "rag_summary.txt")
        return self.state


    @listen("image")
    @router(route_user_input)
    def generate_image(self):
        """Generate AI-powered images based on literary themes and descriptions.
        
        This method handles requests for image generation related to literary content
        by invoking the image generation crew. It creates images based on the
        sanitized user input using AI image generation models.
        
        The method:
        1. Measures execution time for performance tracking
        2. Creates a new ImageCrew instance and executes it
        3. Uses the sanitized input as the topic for image generation
        4. Logs performance metrics to MLflow
        5. Appends the generated image path to the conversation
        
        Returns:
            str: Always returns "new_turn" to continue the conversation flow.
            
        Note:
            The generated image is saved to the filesystem and the path is
            returned to the user interface for display.
        """
        started = time.perf_counter()
        crew = ImageCrew().crew()
        path = crew.kickoff(inputs={'topic': self.state.sanitized_result})

        self.append_agent_response(path.raw, "image")
        duration = time.perf_counter() - started
        self.state.summary = str(path.raw)
 
        # Metriche/artefatti ricerca
        mlflow.log_metric("image_duration_seconds", duration)
        mlflow.log_text(self.state.summary, "image_summary.txt")
        return "new_turn"
    
    @listen(or_(do_rag_and_search, generate_image))
    @router(route_user_input)
    def display_results(self):
        """Display results and perform LLM-as-a-judge evaluation.
        
        This method handles the final stage of processing by displaying the results
        from either RAG search or image generation, and then performing automated
        evaluation using MLflow's LLM-as-a-judge capabilities.
        
        The evaluation process includes:
        - Answer relevance assessment
        - Toxicity detection
        - Faithfulness evaluation (if context is provided)
        - Answer similarity and correctness (if ground truth is provided)
        
        The method automatically configures Azure OpenAI settings and logs
        all evaluation metrics to MLflow for tracking and analysis.
        
        Returns:
            str: Always returns "new_turn" to continue the conversation flow.
            
        Note:
            Evaluation failures are gracefully handled and logged to MLflow
            with appropriate error tags for debugging purposes.
        """
        print("\n📋 RISULTATI DELLA RICERCA WEB\n")
        print(f"🔍 Query: {self.state.user_input}\n")
        print(self.state.summary)

        # --- LLM-as-a-judge con MLflow --- 
        # 🔥 Instead of blocking here, wrap evaluation in a thread
        def background_eval(user_query, prediction, context, ground_truth):
            try:
                os.environ["OPENAI_API_BASE"] = "https://aiacademymainlucamaci.openai.azure.com/"
                os.environ["openai_api_base"] = "https://aiacademymainlucamaci.openai.azure.com/"

                os.environ["OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"
                os.environ["openai_deployment_name"] = "gpt-4o"

                os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
                os.environ["openai_api_version"] = "2024-12-01-preview"

                os.environ["openai_api_type"] = "azure"

                eval_metrics = self._run_llm_judge_mlflow(
                    user_query=user_query,
                    prediction=prediction,
                    context=context or None,
                    ground_truth=ground_truth or None,
                )
                if eval_metrics:
                    mlflow.log_dict(eval_metrics, "eval_metrics_snapshot.json")
                    mlflow.set_tag("llm_judge_status", "success")
            except Exception as e:
                mlflow.set_tag("llm_judge_status", f"failed:{type(e).__name__}")
                mlflow.log_text(str(e), "llm_judge_error.txt")

        # 🔥 Start thread (non-blocking, daemon so it won’t hang shutdown)
        threading.Thread(
            target=background_eval,
            args=(
                self.state.user_input,
                self.state.summary,
                self.state.context,
                self.state.ground_truth,
            ),
            daemon=True
        ).start()

        # 🔥 Immediately return to flow loop instead of waiting
        return "new_turn"
 
 
    # ---------- judge con mlflow.evaluate ----------
    def _run_llm_judge_mlflow(
        self,
        user_query: str,
        prediction: str,
        context: str | None = None,
        ground_truth: str | None = None,
    ):
        """Run LLM-as-a-judge evaluation using MLflow's integrated evaluation metrics.
        
        This method performs automated evaluation of the assistant's responses using
        MLflow's built-in evaluation capabilities. It supports multiple evaluation
        metrics based on the available data:
        
        - answer_relevance: Evaluates how relevant the answer is to the input query
        - toxicity: Detects potentially harmful or toxic content (non-LLM metric)
        - faithfulness: Evaluates answer consistency with provided context
        - answer_similarity: Measures semantic similarity to ground truth
        - answer_correctness: Evaluates factual correctness against ground truth
        
        Args:
            user_query (str): The original user query/question.
            prediction (str): The assistant's response to evaluate.
            context (str | None, optional): Additional context for faithfulness evaluation.
                Defaults to None.
            ground_truth (str | None, optional): Reference answer for similarity/correctness
                evaluation. Defaults to None.
                
        Returns:
            dict: Dictionary containing all computed evaluation metrics.
            
        Note:
            All metrics and evaluation tables are automatically logged to the active
            MLflow run. The method dynamically selects appropriate metrics based on
            available input data (context and ground_truth).
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
 
   
    @listen("quit_flow")
    def quit_flow(self):
        """Handle user quit request and clean up the conversation flow.
        
        This method is triggered when the user requests to quit the conversation.
        It appends a goodbye message to the conversation and terminates the flow.
        
        Returns:
            str: Always returns "end" to terminate the conversation flow.
        """
        print(" Inside quit_flow")
        self.append_agent_response("Goodbye!", "text")
        return "end"
   
 
 
    def append_agent_response(self, response: str | Any, type: Literal["text", "image", "md_file"] = "text"):
        """Append an agent response to the conversation state.
        
        This method adds an assistant response to both the response list and chat history,
        updating conversation counters and refresh flags.
        
        Args:
            response (str | Any): The response content (text, image path, or file path).
            type (Literal["text", "image", "md_file"], optional): Type of response content
                for proper rendering in the UI. Defaults to "text".
        """
        self.state.assistant_response.append(response)
        self.state.chat_history.append({"role": "assistant", "content": response, "type": type})
        self.state.messages_count += 1
        self.state.needs_refresh = True
 
    def append_user_message(self, message: str):
        """Append a user message to the conversation state.
        
        This method adds a user message to the chat history and updates
        conversation counters and refresh flags.
        
        Args:
            message (str): The user's message content.
        """
        self.state.chat_history.append({"role": "user", "content": message})
        self.state.messages_count += 1
        self.state.needs_refresh = True
 
def kickoff():
    """Initialize and start the L_AI_brary chatbot flow with MLflow tracking.
    
    This function sets up the main application entry point by:
    1. Configuring MLflow experiment tracking with tags and metadata
    2. Starting a nested MLflow run for experiment organization
    3. Initializing and launching the ChatbotFlow
    4. Logging execution duration metrics
    
    The function automatically tracks:
    - Application name and flow identification
    - Execution start time in UTC
    - Environment information
    - Total execution duration
    
    Note:
        This function should be called as the main entry point for the application.
        It requires proper MLflow and environment configuration to be set up.
    """
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
    """Generate and display a visual representation of the ChatbotFlow.
    
    This function creates a flow diagram showing the conversation workflow,
    including all states, transitions, and decision points in the ChatbotFlow.
    Useful for debugging, documentation, and understanding the flow structure.
    """
    chatbot_flow = ChatbotFlow()
    chatbot_flow.plot()
 
 
if __name__ == "__main__":
    kickoff()