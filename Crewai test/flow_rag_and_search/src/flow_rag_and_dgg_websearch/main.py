import os
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, router, listen
from flow_rag_and_dgg_websearch.crews.rag_crew.rag_crew import RagCrew
from flow_rag_and_dgg_websearch.crews.search_crew.search_crew import SearchCrew
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS


load_dotenv()
faiss_0__or__qdrant_1 = int(os.environ["FAISS_0__OR__QDRANT_1"])

embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")


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

class RagVsSearchFlow(Flow):

    @start()
    def ask_question(self):
        question = input("Fai la tua richiesta: ")
        return question

    @router(ask_question)
    def route_question(self, question: str):
        print(f"The question is {question}")
        docs = rag_index.similarity_search_with_score(question, k=1)
        best_doc, score = docs[0]
        print(f"Best similarity score: {score:.3f}")
        print(f"With doc: {best_doc}")

        if score < 0.6:
            return "search"
        elif score < 0.8:
            return "both"
        else:
            return "rag"


    @listen("search")
    def run_search(self, question: str):
        print("RAG non adatta a rispondere a questo prompt. Procedo con ricerca web.")
        search_crew = SearchCrew().crew()
        result = search_crew.kickoff(inputs={"query": question})
        print("\n--- Risultato Web ---")
        print(result)
        return result

    @listen("both")
    def run_both(self, question: str):
        print("RAG solo parzialmente adatta a rispondere a questo prompt. Procedo anche con ricerca web.")
        search_crew = SearchCrew().crew()
        rag_crew = RagCrew().crew()
        search_result = search_crew.kickoff(inputs={"query": question})
        rag_result = rag_crew.kickoff(inputs={"query": question})
        print("\n--- Risultato Combinato RAG + WebSearch ---")
        print(f"RAG: {rag_result}\nWEB: {search_result}")
        return {"rag": rag_result, "web": search_result}

    @listen("rag")
    def run_rag(self, question: str):
        print("RAG adatta a rispondere a questo prompt.")
        rag_crew = RagCrew().crew()
        result = rag_crew.kickoff(inputs={"query": question})
        print()
        print("\n--- Risultato RAG ---")
        print(result)
        return result
    
def kickoff():
    flow = RagVsSearchFlow()
    flow.plot("tool_choice_flow")
    flow.kickoff()

if __name__ == "__main__":
    kickoff()
