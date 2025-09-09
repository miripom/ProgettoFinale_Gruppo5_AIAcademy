## Tools e Utilities

### Tools
Percorso: `l_ai_brary/src/l_ai_brary/tools/`

- `rag_tool.py`: funzioni di supporto alla ricerca/indicizzazione per il RAG.
- `image_tool.py`: utilità per generazione e gestione immagini.
- `sanitize_tool.py`: utilità per normalizzazione/sanitizzazione testi.

Questi tools sono orchestrati dai crews corrispondenti. Consultare i file per parametri e API esposte.

### Utilities
Percorso: `l_ai_brary/src/l_ai_brary/utils/`

- `rag_utils.py`: helper per pipeline RAG (embedding, query, post-processing, ecc.).

### Knowledge base
Percorso: `l_ai_brary/src/l_ai_brary/knowledge_base/`

Contiene documenti indicizzati per la ricerca (es. `Short AI Act.pdf`). Assicurarsi che i pipeline di ingest/indicizzazione siano coerenti con il formato atteso dai tools.


