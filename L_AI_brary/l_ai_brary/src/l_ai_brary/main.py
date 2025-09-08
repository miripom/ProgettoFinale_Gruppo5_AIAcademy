from crewai.flow.flow import Flow, start, listen
from l_ai_brary.crews.search_crew.search_crew import SearchCrew

class MainFlow(Flow):
    
    @start()
    def ask_question(self):
        question = input("Fai la tua domanda: ")
        return question
    

    @listen("ask_question")
    def run_search(self, question: str):
        search_crew = SearchCrew().crew()
        result = search_crew.kickoff(inputs={"query": question})
        print("\n--- Risultato Web ---")
        print(result)
        return result

def kickoff():
    flow = MainFlow()
    flow.kickoff()

if __name__ == "__main__":
    kickoff()
