"""RAG evaluation using RAGAS metrics.

This module implements a comprehensive evaluation system for Retrieval-Augmented Generation (RAG)
using the RAGAS framework. It evaluates various aspects of RAG performance including context
precision, context recall, faithfulness, and answer relevancy.

The module builds a dataset from questions and ground truth answers, runs RAG queries,
and evaluates the results using multiple RAGAS metrics to assess the quality of the
retrieval and generation components.
"""

import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

# Aggiungi entrambe le directory al path
sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)
pd.set_option("display.max_columns", None)

from utils.rag_utils import RAG_Settings, hybrid_search, build_rag_chain, get_llm, get_embeddings, format_docs_for_prompt, get_qdrant_client
from typing import List
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)

settings = RAG_Settings()

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY_PATH = os.path.dirname(CURRENT_FILE_PATH)

def build_ragas_dataset(
    questions: List[str],
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """Build a dataset for RAGAS evaluation by running RAG pipeline on questions.
    
    Executes the RAG pipeline for each question and constructs a dataset suitable
    for RAGAS evaluation. Each row contains the question, retrieved contexts,
    generated answer, and optionally the ground truth answer.
    
    Args:
        questions (List[str]): List of questions to evaluate.
        chain: The RAG chain object used to generate answers.
        k (int): Number of top documents to retrieve for context.
        ground_truth (dict[str, str] | None, optional): Dictionary mapping questions
            to their ground truth answers. Defaults to None.
    
    Returns:
        list: List of dictionaries, each containing evaluation data for one question
            with keys: 'user_input', 'retrieved_contexts', 'response', and optionally
            'reference' if ground_truth is provided.
    
    Note:
        The function uses hybrid search to retrieve relevant contexts and formats
        them appropriately for the RAGAS evaluation framework.
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


settings.collection="Short AI Act"
questions = [
            "What is the main purpose of the Artificial Intelligence Act?",

            "How does the AI Act apply a risk-based approach?",

            "Which AI practices are explicitly prohibited under the Act?",

            "What requirements apply to high-risk AI systems?",
            ]

ground_truth = {
                questions[0]: f"The AI Act aims to ensure the smooth functioning of the internal market while protecting fundamental rights, supporting innovation, and fostering trustworthy, human-centric AI in line with EU values",

                questions[1]: f"The AI Act follows a risk-based approach by prohibiting unacceptable AI practices, setting mandatory requirements for high-risk AI systems, and establishing transparency obligations for certain AI applications",

                questions[2]: f"Prohibited practices include manipulative or exploitative techniques that distort human behavior, biometric categorisation revealing sensitive attributes, and social scoring systems that may lead to discrimination or exclusion",

                questions[3]: f"High-risk AI systems must comply with strict requirements to protect health, safety, and fundamental rights, including robust risk management, transparency, documentation, and conformity assessments before being placed on the market."
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
