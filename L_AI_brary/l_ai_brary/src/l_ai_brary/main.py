#!/usr/bin/env python
from random import randint
import time
from typing import Any, Literal

from crewai import LLM
from pydantic import BaseModel

from crewai.flow.flow import Flow, start, router, listen

from l_ai_brary.crews.rag_and_search_crew.rag_and_search_crew import RagAndSearchCrew
from l_ai_brary.crews.image_crew.image_crew import ImageCrew
from l_ai_brary.crews.search_crew.search_crew import SearchCrew
from l_ai_brary.crews.sanitize_crew.sanitize_crew import SanitizeCrew

class ChatState(BaseModel):
    chat_history: list[dict] = []
    user_input: str = ""
    assistant_response: list = [] # might contain text and images. in streamlit we can iterate over it to display them in the order they appear in here
    messages_count: int = 0
    needs_refresh: bool = False  # Flag to indicate UI needs refresh
    user_quit: bool = False
    sanitized_result: str = ""


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

        self.state.sanitized_result = SanitizeCrew().crew().kickoff(inputs={"user_input": query})
        print(" SanitizeCrew result: ", self.state.sanitized_result)

        llm = LLM(model='azure/gpt-4o')
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a binary classifier that routes user queries (after they have been sanitized) to the appropriate service.\n"
                    f"Analyze the sanitized query: '{self.state.sanitized_result}' and answer as follows:\n"
                    f"1 - If the sanitized query is asking to create, generate, draw, or visualize something, return 'image';\n"
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
            self.append_agent_response(self.state.sanitized_result)
            return "image"
        elif classification.lower() == "rag_and_search":
            self.append_agent_response(self.state.sanitized_result)
            return "rag_and_search"
        else:
            self.append_agent_response(self.state.sanitized_result)
            return "new_turn"


    @listen("rag_and_search")
    @router(route_user_input)
    def do_rag_and_search(self):
        # call RAG crew
        # crew = RagAndSearchCrew().crew()
        # answer = crew.kickoff(inputs={'input': self.state.sanitized_result})
        # print(" Inside do_rag_and_search")
        # self.append_agent_response(answer, "text")
        return "new_turn"


    @listen("image")
    @router(route_user_input)
    def generate_image(self):
        crew = ImageCrew().crew()
        path = crew.kickoff(inputs={'topic': self.state.sanitized_result.raw})
        
        self.append_agent_response(path.raw, "image") # TODO: change to "image" when image generation is implemented
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
