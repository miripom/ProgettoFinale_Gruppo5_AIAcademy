#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from l_ai_brary.crews.pdf_crew.pdf_crew import PdfCrew


class ChatState(BaseModel):
    chat_history: list[dict] = []
    messages_count: int = 0
    user_quit: bool = False


class ChatbotFlow(Flow[ChatState]):

    @start()
    def take_user_input(self, user_input: str | None = None):
        if self.state.user_quit:
            return None
        
        if user_input:
            # Append user message
            self.state.chat_history.append({"role": "user", "content": user_input})
            self.state.messages_count += 1

            # Dummy assistant response (replace with crew orchestration)
            assistant_response = f"Echo: {user_input}"
            self.state.chat_history.append({"role": "assistant", "content": assistant_response})
        
        return self.state


def kickoff():
    chatbot_flow = ChatbotFlow()
    chatbot_flow.kickoff()


def plot():
    chatbot_flow = ChatbotFlow()
    chatbot_flow.plot()


if __name__ == "__main__":
    kickoff()
