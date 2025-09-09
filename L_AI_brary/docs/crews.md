## Crews

I crews sono definiti in `l_ai_brary/src/l_ai_brary/crews/*` con configurazioni YAML in `config/agents.yaml` e `config/tasks.yaml`.

### SanitizeCrew
- Percorso: `crews/sanitize_crew`
- Scopo: sanitizzare l'input utente per evitare injection e contenuti non sicuri.
- Uso nel flow: chiamato in `route_user_input`, risultato salvato in `state.sanitized_result`.

### RagAndSearchCrew
- Percorso: `crews/rag_and_search_crew`
- Scopo: eseguire ricerca semantica e RAG sulla knowledge base (`knowledge_base/`).
- Output: testo riassuntivo/risposta, loggato in MLflow come artefatto `search_summary.txt`.
- Uso nel flow: `do_rag_and_search` con input `{"query": state.user_input}`.

### ImageCrew
- Percorso: `crews/image_crew`
- Scopo: generare immagini a tema letterario a partire dall'input sanitizzato.
- Output: path dell'immagine generata, aggiunto a `assistant_response` con tipo `image`.
- Uso nel flow: `generate_image` con input `{"topic": state.sanitized_result}`.

### Configurazioni
Le configurazioni in YAML definiscono agenti e task; assicurarsi che i riferimenti a modelli, chiavi e parametri siano coerenti con le variabili d'ambiente e i servizi disponibili (Azure OpenAI, ecc.).


