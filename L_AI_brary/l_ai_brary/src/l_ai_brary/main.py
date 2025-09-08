#!/usr/bin/env python
from random import randint
import time
from typing import Any, Literal

from pydantic import BaseModel

from crewai.flow.flow import Flow, start, router, listen

from l_ai_brary.crews.pdf_crew.pdf_crew import PdfCrew

class ChatState(BaseModel):
    chat_history: list[dict] = []
    user_input: str = ""
    assistant_response: list = [] # might contain text and images. in streamlit we can iterate over it to display them in the order they appear in here
    messages_count: int = 0
    needs_refresh: bool = False  # Flag to indicate UI needs refresh
    user_quit: bool = False


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
        
        return self.state

    @router(wait_for_input)
    def route_user_input(self):
        """Decide what to do with the user query."""
        print(" Inside route_user_input")
        print(f"with query: {self.state.user_input}")
        query = self.state.user_input
        if not query:
            time.sleep(2)
            return "new_turn"  # loop back until user says something
        # Simple routing logic — later can be replaced by an LLM-based router
        if "image" in query.lower():
            return "image"
        elif "rag" in query.lower() or "book" in query.lower():
            return "rag"
        elif "websearch" in query.lower():
            return "websearch"
        else:
            return "new_turn"

    @listen("rag")
    @router(route_user_input)
    def do_rag(self):
        # call RAG crew
        print(" Inside do_rag")
        self.append_agent_response("[RAG Answer]", "text")
        return "new_turn"

    @listen("websearch")
    @router(route_user_input)
    def do_websearch(self):
        # call websearch crew
        print(" Inside do_websearch")
        self.append_agent_response("[Websearch Answer]", "text")
        return "new_turn"

    @listen("image")
    @router(route_user_input)
    def generate_image(self):
        print(" Inside generate_image")
        self.append_agent_response("[Generated Image]", "image")
        return "new_turn"

    @listen("clarification")
    @router(route_user_input)
    def ask_for_clarification(self):
        print(" Inside ask_for_clarification")
        self.append_agent_response("Please clarify your request.", "text")
        return "new_turn"


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
    chatbot_flow = ChatbotFlow()
    chatbot_flow.kickoff()


def plot():
    chatbot_flow = ChatbotFlow()
    chatbot_flow.plot()


if __name__ == "__main__":
    kickoff()
