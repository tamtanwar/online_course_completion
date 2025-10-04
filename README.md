🎓 Online Course Completion Prediction

Predict whether a student will complete an online course using ML and real engagement data. FastAPI API included for real-time predictions. Fully Dockerized.


🚀 Features
	•	✅ Predict course completion (Yes / No) with probability
	•	🔄 Preprocessing & feature scaling included
	•	🧮 Baseline model: Logistic Regression
	•	⚡ FastAPI /predict endpoint for real-time predictions
	•	🐳 Dockerized for easy deployment


🛠 Tech Stack

Python 3.12+, Pandas, NumPy, Scikit-learn, FastAPI, Uvicorn, Joblib

Quick Start
	1.	Clone repo
git clone <repo-url>
cd mlproject/task1
	2.	Setup environment
python -m venv mlproject_env
source mlproject_env/bin/activate (macOS/Linux)
mlproject_env\Scripts\activate (Windows)
pip install fastapi uvicorn pandas scikit-learn joblib
	3.	Run API locally
python main.py

	•	Root: http://localhost:8000/
	•	Swagger docs: http://localhost:8000/docs

	4.	Example /predict request
{"age": 25, "hours_per_week": 10, "country": "USA"}
Response: {"prediction": 1, "probability": 0.99966}

⸻

Docker (Optional)
	•	Build: docker build -t online-course-api .
	•	Run: docker run -p 8000:8000 online-course-api

⸻

Optional Model Training

python app/train_model.py --data data/online_course.csv --model_path saved_models/baseline_logreg.joblib
