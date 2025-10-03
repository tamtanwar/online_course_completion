from app.inference import InferenceModel

# Path to your saved model
model_path = "/Users/tamanna/Documents/ML project /Task 1/saved_models/baseline_logreg.joblib"  # update if needed

# Initialize inference model
model = InferenceModel(model_path=model_path)

# Example input — can include only a subset of features
student = {
    "age": 23,
    "hours_per_week": 10,
    "country": "India",
    # other features can be omitted; they will be auto-filled
}

# Run prediction
result = model.predict(student)

print("Prediction result:", result)