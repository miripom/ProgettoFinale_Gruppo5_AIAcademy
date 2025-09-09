# ProgettoFinale_Gruppo5_AIAcademy

Repository per il progetto finale del gruppo 5 per l'AI Academy

How to run:


1.
Start qdrant server on http://localhost:6333
and access it via http://localhost:6333/dashboard
docker run -d -p 6333:6333 qdrant/qdrant

2.
Start mlflow backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

3.
Go to 
cd "L_AI_brary\l_ai_brary\src\l_ai_brary"  

4.
Activate the .venv
.venv\Scripts\Activate

5.
Generate the streamlit frontend on http://localhost:8501/ :
python -m streamlit run streamlit_frontend/app.py