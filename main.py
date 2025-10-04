from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import InferenceModel

# Initialize FastAPI app
app = FastAPI(title="Online Course Completion Prediction API")

# Load all models
baseline_model = InferenceModel("saved_models/baseline_logreg.pkl")
rf_model = InferenceModel("saved_models/random_forest_v1.pkl")
gb_model = InferenceModel("saved_models/gradient_boosting_v1.pkl")

# Define input schema
class StudentInput(BaseModel):
    age: int
    hours_per_week: float
    country: str
    # add all other features your model expects, e.g.:
    # education_level: str
    # num_logins_last_month: int
    # videos_watched_pct: float
    # assignments_submitted: int
    # discussion_posts: int
    # is_working_professional: int
    # preferred_device: str
    # weight_kg: float
    # height_cm: float

@app.get("/")
def root():
    return {"message": "Online Course Completion Prediction API is running!"}

@app.post("/predict")
def predict(student: StudentInput, model_type: str = "baseline"):
    # Convert input to dict
    input_data = student.dict()
    
    # Choose model
    if model_type.lower() == "baseline":
        model = baseline_model
    elif model_type.lower() == "random_forest":
        model = rf_model
    elif model_type.lower() == "gradient_boosting":
        model = gb_model
    else:
        return {"error": "Invalid model_type. Choose 'baseline', 'random_forest', or 'gradient_boosting'"}
    
    # Make prediction
    result = model.predict(input_data)
    
    return {"model_type": model_type, "prediction": result["prediction"], "probability": result["probability"]}