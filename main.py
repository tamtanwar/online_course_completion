# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import InferenceModel
import uvicorn
import os

# -------------------------------
# 1. Define input schema
# -------------------------------
class StudentData(BaseModel):
    age: int
    hours_per_week: float
    country: str
    # Add any other features your model expects
    # Example:
    # assignments_submitted: int
    # favorite_color: str
    # num_logins_last_month: int
    # ...

# -------------------------------
# 2. Initialize FastAPI app
# -------------------------------
app = FastAPI(
    title="Online Course Completion Prediction API",
    description="Send student features and get course completion prediction",
    version="1.0"
)

# -------------------------------
# 3. Load inference model
# -------------------------------
# Make sure the path points to your saved model
model_path = os.path.join("saved_models", "baseline_logreg.joblib")
model = InferenceModel(model_path=model_path)

# -------------------------------
# 4. Define endpoints
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome! Use /predict endpoint to get predictions."}

@app.post("/predict")
def predict(student: StudentData):
    # Convert Pydantic model to dictionary
    student_dict = student.dict()
    
    # Run inference
    result = model.predict(student_dict)
    
    return {
        "prediction": result["prediction"],
        "probability": result["probability"]
    }

# -------------------------------
# 5. Run the server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)