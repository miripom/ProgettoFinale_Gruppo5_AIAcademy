## Setup

Questa guida spiega come avviare L_AI_brary in locale.

### Prerequisiti
- Python 3.11
- Virtualenv o venv
- Docker (per Qdrant)
- Variabili d'ambiente per Azure OpenAI/Qdrant/MLflow (vedi `.env`)

### 1) Clona e posizionati nella repo
```bash
cd L_AI_brary
```

### 2) Avvia Qdrant via Docker
```bash
docker run -d -p 6333:6333 qdrant/qdrant
# Dashboard: http://localhost:6333/dashboard
```

### 3) Attiva l'ambiente virtuale
Da Windows PowerShell:
```powershell
cd l_ai_brary\l_ai_brary\src\l_ai_brary
..\..\..\..\myenv\Scripts\Activate.ps1
```
Oppure, se usi un venv locale:
```powershell
.venv\Scripts\Activate
```

### 4) Configura MLflow
Il codice imposta automaticamente `MLFLOW_TRACKING_URI` di default a `http://127.0.0.1:5001`.
Se utilizzi un server diverso, imposta la variabile d'ambiente:
```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
```

### 5) Variabili d'ambiente Azure OpenAI
Imposta nel tuo `.env` (caricato da `dotenv`):
```
OPENAI_API_BASE=https://<your-azure-openai>.openai.azure.com/
OPENAI_API_VERSION=2024-12-01-preview
OPENAI_DEPLOYMENT_NAME=gpt-4o
OPENAI_API_KEY=<your-key>
```

### 6) Avvio frontend Streamlit
```powershell
python -m streamlit run streamlit_frontend/app.py
# App su http://localhost:8501
```

### 7) Esecuzione del flow (opzionale)
Il main flow è definito in `l_ai_brary/src/l_ai_brary/main.py` (funzione `kickoff`).
Puoi integrarlo nel frontend o usarlo programmaticamente.


