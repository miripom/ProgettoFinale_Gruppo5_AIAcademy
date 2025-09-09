## Architettura e Flow

L_AI_brary è organizzato attorno a un CrewAI Flow (`ChatbotFlow`) che orchestra tre crews specializzati, con stato centralizzato gestito da `ChatState`.

### Componenti principali
- `ChatState`: stato conversazionale con chat history, input utente, risposte, flag UI, risultati sanitizzazione, riassunto/risposta, contesto e ground truth per valutazione. Contiene istanze condivise di:
  - `LLMclassifier` (Azure GPT-4o)
  - `SanitizeCrew`
  - `ImageCrew`
  - `RagAndSearchCrew`

- `ChatbotFlow`: gestisce il ciclo di vita della conversazione con fasi/stati:
  - `wait_for_input` (start): attende input, pulizia stato del turno, logging MLflow di base
  - `route_user_input` (router): sanitizza input, classifica con LLM in `image | rag_and_search | new_turn`
  - `do_rag_and_search` (listen): esegue RAG+search, logga metriche e testo
  - `generate_image` (listen): genera immagine basata su input sanitizzato, logga metriche e path
  - `display_results` (listen): mostra risultati e lancia valutazione LLM-as-a-judge con MLflow
  - `quit_flow` (listen): termina il flow

### Routing logico
1. L'utente inserisce testo → `wait_for_input`
2. `route_user_input` invoca `SanitizeCrew` e poi classifica:
   - "image" → `generate_image`
   - "rag_and_search" → `do_rag_and_search`
   - default → risposta con testo sanitizzato e ritorno a `new_turn`

### Logging e telemetria
- MLflow: `mlflow.autolog()` e metriche custom (durata, lunghezza risposta, numero righe), artefatti (`search_summary.txt`, snapshot metriche valutazione)
- Tag run: nome app/flow, timestamp, ambiente

### Valutazione (LLM-as-a-judge)
In `display_results` si configurano variabili Azure OpenAI e si chiama `_run_llm_judge_mlflow`, che usa `mlflow.evaluate` con metriche:
- `answer_relevance`, `toxicity`
- `faithfulness` se presente `context`
- `answer_similarity`, `answer_correctness` se presente `ground_truth`


