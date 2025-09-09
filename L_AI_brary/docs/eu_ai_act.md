## EU AI Act Compliance Documentation — L_AI_brary

- **Application Owners**: Luca Maci — email: luca.maci@it.ey.com, Miriana Pompilio - email: miriana.pompilio@it.ey.com, Pasquale Fidanza - email: pasquale.fidanza@it.ey.com
- **Document Version**: 0.1 — Data: 2025-09-09


### Key Links
- **Code Repository**: `https://github.com/miripom/ProgettoFinale_Gruppo5_AIAcademy/tree/main`
- **API (Swagger Docs)**: Nessuna API presente
- **Cloud Account**: Esecuzione locale
- **Application Architecture**: `docs/architecture.md`

---

## General Information

Riferimenti normativi: **EU AI Act Art. 11; Annex IV §1,2,3**

### Purpose and Intended Use
- **Descrizione**: L_AI_brary è un assistente letterario che combina sanitizzazione dell’input, RAG (Retrieval-Augmented Generation) su una knowledge base letteraria, generazione di immagini a tema e valutazione automatica LLM-as-a-judge, orchestrati tramite CrewAI e tracciati con MLflow.
- **Problema affrontato**: Supportare utenti nella ricerca e sintesi di contenuti letterari, generare immagini ispirate a temi/descrizioni letterarie, e mantenere sicurezza dei contenuti tramite sanitizzazione.
- **Settore**: Applicazioni consumer/educational/culturali (information retrieval e generazione creativa).
- **Utenti e stakeholder**: Utenti finali, owner del progetto, revisori di compliance, team ML/engineers.
- **Obiettivi e KPI**:
  - Qualità risposte RAG (answer_relevance > soglia definita; faithfulness quando disponibile contesto)
  - Sicurezza contenuti (toxicity sotto soglia)
  - Performance applicativa (latenza per query; uptime)
  - Tracciabilità ML (run e metriche in MLflow)
- **Implicazioni etiche e vincoli**: Minimizzare allucinazioni, bias, e contenuti inappropriati; rispetto di licenze dei dati; trasparenza sui limiti del sistema.
- **Usi vietati / misuse**: Non usare per decisioni ad impatto legale, sanitario, di sicurezza o occupazionale. Non utilizzare per generare disinformazione o contenuti illeciti.
- **Ambiente operativo**: Esecuzione locale con frontend Streamlit; integrazione con Azure OpenAI (LLM), Qdrant (DB vettoriale), MLflow (tracking). Possibile estensione cloud.

### Risk classification
- Riferimenti: **Prohibited Risk (Art. 5)**; **High-Risk (Art. 6-7)**; **Limited Risk (Art. 50)**.
- **Classificazione proposta**: Limited/Minimal risk.
- **Motivazione**: Sistema informativo/creativo senza funzioni critiche in ambiti regolati ad alto rischio; non effettua profilazione biometrica o decisioni automatizzate che producano effetti legali o similmente significativi sugli utenti.

---

## Application Functionality
Riferimenti: **Art. 11; Annex IV §1,2,3** — **Instructions for Use (Art. 13)**

### Istruzioni per i deployer
- Configurare variabili d’ambiente (Azure OpenAI, MLflow, Qdrant). Avviare Qdrant (`docker run -d -p 6333:6333 qdrant/qdrant`). Eseguire frontend: `python -m streamlit run streamlit_frontend/app.py`.
- Garantire che la knowledge base sia adeguata e priva di contenuti non conformi alle licenze.
- Monitorare metriche in MLflow e conservare i log.

### Model Capabilities
- **Può**: Sanitizzare input; eseguire RAG su knowledge base; generare immagini; valutare risposte con metriche (answer_relevance, toxicity, ecc.).
- **Non può/limiti**: Conoscenza limitata alla knowledge base e al modello LLM; possibili allucinazioni; assenza di garanzia di accuratezza fattuale completa; nessuna presa di decisioni automatica vincolante.
- **Lingue**: Italiano/inglese (in base al supporto del modello LLM).

### Input Data Requirements
- **Formato**: Testo libero per query; prompt testuale per immagini.
- **Qualità**: Evitare prompt malevoli; fornire richieste contestualizzate. 
- **Esempi**: 
  - Valido: “Riassumi il capitolo 1 del libro X dalla knowledge base.”
  - Non valido: prompt di injection o contenuti illeciti.

### Output Explanation
- **Interpretazione**: Testo riassuntivo o informativo per RAG; percorso/anteprima immagine per generazione.
- **Incertezza**: Metriche in MLflow; possibilità di aggiungere confidence/uncertainty future.

### System Architecture Overview
- **Descrizione funzionale**: Flow CrewAI (`ChatbotFlow`) con stati: input → sanitizzazione/classificazione → RAG o image → display+valutazione.
- **Componenti**: 
  - Dati: `knowledge_base` e indice vettoriale (Qdrant)
  - Algoritmi/Modelli: Azure OpenAI (LLM/embeddings), pipeline RAG, generazione immagini
  - Orchestrazione: CrewAI (`SanitizeCrew`, `RagAndSearchCrew`, `ImageCrew`)
  - Tracking: MLflow (metriche, artefatti)
  - Frontend: Streamlit

---

## Models and Datasets
Riferimenti: **Art. 11; Annex IV §2(d)**

### Models
| Model | Single Source of Truth | Descrizione uso |
|---|---|---|
| Azure OpenAI `gpt-4o` | Config ambiente (Azure) | Classificazione input, generazione testo, valutazione LLM-as-a-judge |
| Embeddings Azure | Config ambiente (Azure) | Indicizzazione/ricerca semantica RAG |
| Generazione immagini | Config Crew/Tool | Creazione immagini a tema letterario |

### Datasets
| Dataset | Single Source of Truth | Descrizione uso |
|---|---|---|
| Knowledge base locale | `l_ai_brary/src/l_ai_brary/knowledge_base/` | Corpus letterario per RAG |
| Artefatti MLflow | `mlartifacts/` | Tracciamento metriche/risultati |

Note: Documentazione dataset, licenze e fonti devono essere mantenute e referenziate nel repo/dataset cards.

---

## Deployment

### Infrastructure and Environment Details
- **Cloud Setup**: Attuale locale. Estensione possibile su cloud (Azure/GCP/AWS). Definire regioni, servizi (compute, storage, DB), e GPU se necessarie per immagini.
- **Rete**: Config locale; in cloud prevedere VNet/VPC, subnet, SG/NSG.
- **Database vettoriale**: Qdrant su Docker (localhost:6333).

### APIs
- **Frontend**: Streamlit (no Swagger). Eventuali API future: definire endpoint, payload, auth (API key/OAuth).
- **Scalabilità/Latenza**: Dipende da LLM esterno e DB vettoriale; prevedere caching e batching.

### Integration with External Systems
Riferimenti: **Art. 11; Annex IV §1(b,c,d,g,h), §2(a)**
- **Dipendenze**: Azure OpenAI, Qdrant, MLflow, CrewAI.
- **Flussi dati**: Utente → Streamlit → Flow → Sanitizzazione → Classificazione → (RAG con Qdrant/Embeddings) o (Image) → Output → MLflow logging.
- **Error handling**: Try/except in fase di valutazione; aggiungere retry/backoff per chiamate LLM/DB in produzione.

### Deployment Plan
- **Ambienti**: dev → staging → prod (da definire). Oggi: locale.
- **Scaling**: Auto-scaling/app service o container orchestrator in cloud.
- **Backup/Recovery**: Snapshot volume Qdrant; backup artefatti MLflow; export run.
- **Integrazione**: Ordine: DB/indice → variabili ambiente → servizi LLM → app → frontend.
- **Dipendenze**: Librerie Python (CrewAI, MLflow, qdrant-client, dotenv), Docker.
- **Rollback**: Versionare immagini/container e config; feature flags.
- **User information**: Frontend Streamlit su `http://localhost:8501`.

---

## Lifecycle Management
Riferimenti: **Art. 11; Annex IV §6**

### Monitoring
- **Performance app**: latenza, error rate.
- **Performance modello**: answer_relevance, toxicity, faithfulness se disponibile contesto.
- **Infra**: CPU, memoria, rete.

### Versioning & Changelog
- Versionare dataset/indice, config LLM, crews e tools; mantenere CHANGELOG e tag MLflow.

### Key Activities
- Monitoraggio in produzione, gestione drift, bugfix, aggiornamenti periodici.

### Documentation Needs
- **Monitoring logs** (MLflow + applicativi)
- **Incident reports**
- **Retraining/Indexing logs**
- **Audit trails** (MLflow run, commit repo)
- **Change log**: nuove feature, aggiornamenti, deprecazioni, rimozioni, bugfix, security fixes

---

## Risk Management System
Riferimenti: **Art. 9; Art. 11; Annex IV**

### Risk Assessment Methodology
- Ispirato a ISO 31000 / NIST RAF / FMEA light: identificazione, analisi, valutazione, mitigazione, monitoraggio.

### Identified Risks
- **Bias/Allucinazioni**: risposte non accurate o parziali.
- **Privacy/licenze**: uso improprio di contenuti protetti.
- **Sicurezza contenuti**: prompt injection, contenuti tossici.
- **Affidabilità**: downtime LLM o Qdrant.

### Likelihood & Severity (qualitativa)
- Bias/Allucinazioni: medio/medio; Sicurezza contenuti: medio/medio; Affidabilità: medio/medio.

### Risk Mitigation Measures
- **Preventive**: sanitizzazione input; validazione prompt; controllo fonti dataset; metriche MLflow; guardrail nei crew.
- **Protective**: fallback su risposte conservative; degrado controllato; avvisi all’utente; logging incidenti.

---

## Testing and Validation (Accuracy, Robustness, Cybersecurity)
Riferimenti: **Art. 15**

### Accuracy
- **Metriche**: answer_relevance, toxicity; opzionali: faithfulness, answer_similarity/correctness se presenti context/ground truth.
- **Risultati di validazione**: salvati in MLflow (`eval_metrics_snapshot.json`, tabelle evaluate).
- **Misure**: qualità dati, ottimizzazione pipeline RAG, monitoraggio continuo.
- **Ciclo di vita**: data validation, aggiornamento KB/indice, verifica periodica metriche.

### Robustness
- **Analisi outlier**: controllo input anomali; filtri sulla KB.
- **Post-analisi**: ispezione errori, casi limite, regressioni.
- **Misure**: stress test su input lunghi/rumorosi, fallback rules, timeouts e retry.
- **Scenario-based testing**: edge cases (assenza contesto, domande ambigue), degrado graduale.
- **Ridondanza/Failsafe**: risposte conservative e messaggi di errore chiari.
- **Incertezza**: possibilità di esporre confidence/flags in futuro.

### Cybersecurity
Riferimenti: **Annex IV §2(h)**
- **Data Security**: separazione secret (.env), TLS in cloud, principle of least privilege.
- **Access Control**: chiavi API Azure/OpenAI segrete, permessi su MLflow/Qdrant.
- **Incident Response**: runbook: isolare credenziali compromesse, ruotare chiavi, audit MLflow, rollback versione.
- **Post-deployment**: patching dipendenze, log forense, monitor alerting.

---

## Human Oversight
Riferimenti: **Art. 11; Annex IV §2(e); Art. 14**
- **Human-in-the-loop**: l’operatore revisiona gli output; l’utente può richiedere nuova generazione o interrompere.
- **Override/Intervento**: kill switch dell’app; routing a “new_turn” per richieste non pertinenti; fallback testuale.
- **Istruzioni/Training**: guida in `docs/`; convenzioni prompt e best practice; limiti dichiarati.
- **Limitazioni/Constraint**: non usare per decisioni critiche; potenziali errori fattuali; copertura dominio limitata.

---

## Incident Management
- **Common issues**: mancanza variabili d’ambiente, Qdrant non avviato, credenziali Azure non valide.
- **Log e debug**: consultare MLflow, console Streamlit, log applicativi.
- **Supporto**: Owner del progetto, canali del team.

### Troubleshooting Deployment
- **Insufficient Resources**: stimare carichi, vertical/horizontal scaling, caching.
- **Network Failures**: health check, retry/backoff, timeout.
- **Pipeline Failures**: validazione env, lock delle versioni, test pre-deploy.
- **API Failures**: circuit breaker, osservabilità, rotazione chiavi.
- **Data Format Mismatches**: schemi validati, controlli input, test end-to-end.
- **Data Quality**: pulizia/validazione KB, revisioni periodiche.
- **Model Drift/Domain Gap**: monitor metriche, aggiornare KB/indice.
- **Security**: segreti gestiti, scansioni dipendenze, policy accesso.
- **Monitoring/Logging**: definire SLO, allarmi, dashboard MLflow.
- **Rollback**: versioni container/config, snapshot indice.
- **Disaster Recovery**: backup artefatti/KB, ripristino documentato.

---

## EU Declaration of Conformity
Riferimenti: **Art. 47**
- **Standards applied**: ISO 31000 (risk management, ispirazione), NIST AI RMF, best practice MLOps (MLflow).
- Stato: bozza informativa non vincolante (non high-risk).

---

## Documentation Metadata
- **Template Version**: 0.1
- **Authors**: Luca Maci, Team AI&Data (Owner); Miriana Pompilio, Team AI&Data (Owner); Pasquale Fidanza, Team AI&Data (Owner)
