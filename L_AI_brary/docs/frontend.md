## Frontend Streamlit

Percorso app: `l_ai_brary/src/l_ai_brary/streamlit_frontend/app.py`

### Avvio
```powershell
python -m streamlit run streamlit_frontend/app.py
# http://localhost:8501
```

### Funzionalità UI
- Campo testo per input utente
- Visualizzazione chat history (utente/assistente)
- Rendering risposte testuali e immagini nell'ordine generato dal flow (`assistant_response`)
- Pulsante/flag per terminare la conversazione (`user_quit`)

### Integrazione col Flow
L'app imposta/legge `ChatState`:
- `user_input` viene aggiornato dal frontend; il flow lo consuma
- `assistant_response` viene iterato per mostrare testi/immagini
- `needs_refresh` può essere usato per triggerare rerender


