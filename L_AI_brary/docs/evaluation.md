## Valutazione e MLflow

Il progetto integra MLflow per tracking ed evaluation delle risposte del sistema.

### Tracking
- `mlflow.set_tracking_uri` (default `http://127.0.0.1:5001`)
- `mlflow.autolog()` abilita logging automatico dove supportato
- Tag del run: `app_name`, `flow_name`, `run_started_at_utc`, `environment`
- Metriche custom:
  - `user_query_length_chars`, `user_query_length_words`
  - `search_duration_seconds`, `search_results_chars`, `search_results_words`, `search_results_lines`
- Artefatti:
  - `search_summary.txt`
  - `eval_metrics_snapshot.json`

### LLM-as-a-judge
In `display_results` viene invocato `_run_llm_judge_mlflow`, che usa `mlflow.evaluate` su un DataFrame a 1 riga con colonne disponibili: `inputs`, `predictions`, opzionalmente `context` e `ground_truth`.

Metriche utilizzate:
- `mlflow.metrics.genai.answer_relevance()`
- `mlflow.metrics.toxicity()`
- `mlflow.metrics.genai.faithfulness(context_column="context")` se `context` presente
- `mlflow.metrics.genai.answer_similarity()` e `answer_correctness()` se `ground_truth` presente

`model_type` è "question-answering" se c'è `ground_truth`, altrimenti "text".


