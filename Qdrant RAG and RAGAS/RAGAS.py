import os
import pandas as pd
pd.set_option("display.max_columns", None)

from utils.utils import Settings, hybrid_search, build_rag_chain, get_llm, get_embeddings, format_docs_for_prompt, get_qdrant_client
from typing import List
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)

settings = Settings()

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY_PATH = os.path.dirname(CURRENT_FILE_PATH)

def build_ragas_dataset(
    questions: List[str],
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
    """
    dataset = []
    for q in questions:
        hits = hybrid_search(client, settings, q, embeddings)
        contexts = []
        for hit in hits:
            contexts.append(format_docs_for_prompt([hit]))
        # contexts = [format_docs_for_prompt(hit) for hit in hits]
        # print(contexts)
        answer = chain.invoke(q)

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


settings.collection="EU AI Act.pdf"
questions = [
            "Does the EU AI Act apply to AI systems developed outside the EU but used within the EU?",

            "Which AI systems used in law enforcement are classified as high-risk under the EU AI Act, and why?",

            "Who supervises and enforces obligations on providers of general-purpose AI models under the EU AI Act, and how is this done",

            "What happens when the scientific panel issues a qualified alert under Article 90 of the EU AI Act?",
            ]

ground_truth = {
                questions[0]: f"Yes. It applies to providers and users in the EU, even if the system was developed elsewhere (Art. 2).",

                questions[1]: f"AI systems used by or on behalf of law enforcement are classified as high-risk because they can affect fundamental rights such as liberty, non-discrimination, fair trial, and presumption of innocence. High-risk systems include tools like polygraphs, systems assessing a person’s risk of becoming a victim or reoffending, systems for profiling during investigations, and tools for evaluating the reliability of evidence. These require high accuracy, reliability, and transparency to avoid discrimination, wrongful outcomes, and erosion of public trust.",

                questions[2]: f"Supervision and enforcement of obligations on providers of general-purpose AI models fall under the competence of the European Commission, through the AI Office. The AI Office can monitor compliance, investigate infringements on its own or at the request of market surveillance authorities, and receive complaints from downstream providers about possible rule violations.",

                questions[3]: f"If the scientific panel suspects a general-purpose AI model poses a concrete risk at Union level or meets the conditions of Article 51, it may issue a qualified alert to the AI Office. The Commission, via the AI Office, can then assess the matter and may exercise its powers under Articles 91–94, informing the Board. The alert must be reasoned and include the provider’s contact, relevant facts, reasons for concern, and any other relevant information."
              }


client = get_qdrant_client(settings)
embeddings = get_embeddings(settings)
llm = get_llm(settings)
chain = build_rag_chain(llm)


dataset = build_ragas_dataset(
    questions=questions,
    chain=chain,
    k=settings.final_k,
    ground_truth=ground_truth,  # rimuovi se non vuoi correctness
)
evaluation_dataset = EvaluationDataset.from_list(dataset)
metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

if all("ground_truth" in row for row in dataset):
    metrics.append(answer_correctness)

# Esegui la valutazione con il TUO LLM e le TUE embeddings
ragas_result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
    embeddings=embeddings,  # o riusa 'embeddings' creato sopra
)

df = ragas_result.to_pandas()
cols = ["user_input", "response", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
print("\n=== DETTAGLIO PER ESEMPIO ===")
print(df[cols].round(4).to_string(index=False))

# (facoltativo) salva per revisione umana
df.to_csv(f"{CURRENT_DIRECTORY_PATH}/ragas_results.csv", index=False)
print("Salvato: ragas_results.csv")

df = pd.read_csv(f"{CURRENT_DIRECTORY_PATH}/ragas_results.csv")

print(df)
print(f'{"-"*50} RETRIEVED CONTEXTS {"-"*50}')
for i, val in enumerate(df["retrieved_contexts"]):
  print(f"Row {i}: {val}\n")

print(f'{"-"*50} RESPONSES {"-"*50}')
for i, val in enumerate(df["response"]):
  print(f"Row {i}: {val}\n")
