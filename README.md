Online Course Completion Prediction

Project Overview

This project predicts whether a student will complete an online course based on features like age, study hours, and other behavioral metrics.
We built a binary classification ML model and wrapped it in a FastAPI API, then containerized it using Docker for easy deployment.

⸻

Features
	•	Predict course completion: Yes / No
	•	Returns probability of completion
	•	API is modular and production-ready
	•	Fully containerized with Docker

Project Structure
mlproject/
│
├── task1/
│   ├── main.py                # FastAPI application
│   ├── Dockerfile             # Docker configuration
│   ├── saved_models/
│   │   └── baseline_logreg.joblib   # Trained ML model
│   └── app/
│       ├── inference.py       # InferenceModel class for predictions
│       └── train_model.py     # Training script (optional)
Getting Started

1. Clone the repo
2. git clone <your-repo-url>
cd mlproject/task1
2. Create virtual environment (optional)
   python -m venv mlproject_env
source mlproject_env/bin/activate   # macOS/Linux
# or
mlproject_env\Scripts\activate      # Windows
3. Install dependencies
pip install fastapi uvicorn pandas scikit-learn joblib
Running the API Locally
python main.py
	•	Root endpoint: http://localhost:8000/
	•	Swagger docs: http://localhost:8000/docs
	•	Test /predict by sending JSON input like:
{
  "age": 25,
  "hours_per_week": 10,
  "country": "USA"
}
	•	Response example:
{
  "prediction": 1,
  "probability": 0.99966
}
Docker Setup

1. Build Docker image
docker build -t online-course-api .
2. Run Docker container
docker run -p 8000:8000 online-course-api
	•	API is now accessible at http://localhost:8000/
	•	Swagger docs: http://localhost:8000/docs
3. Stop container
docker ps            # find container ID
docker stop <id>
Model Training
	•	The model is trained using scikit-learn Logistic Regression (baseline).
	•	Preprocessing (scaling, encoding) is handled inside InferenceModel to ensure consistent predictions.
	•	You can optionally train your own model using train_model.py:
python app/train_model.py --data data/online_course.csv --model_path saved_models/baseline_logreg.joblib

Key Notes
	•	Dependency Versions:
	•	Python 3.12
	•	scikit-learn 1.5.1
	•	pandas 2.2.2
	•	numpy 1.26.4
	•	Versions must match when loading saved models to avoid errors.

⸻
Project Workflow
	1.	EDA & Preprocessing: Understand dataset, clean, encode features.
	2.	Train Baseline Model: Logistic Regression for first benchmark.
	3.	Create Inference Class: Modular predictions.
	4.	FastAPI API: Expose /predict endpoint.
	5.	Dockerize: Make API portable & deployable anywhere.

⸻
Future Improvements
	•	Add more ML models (Random Forest, XGBoost) for better accuracy
	•	Deploy API to cloud (AWS / GCP / Azure)
	•	Add authentication for secure API access
