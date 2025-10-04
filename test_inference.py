from app.inference import InferenceModel

# --------------------------
# Paths to saved models
baseline_model_path = "saved_models/baseline_logreg.pkl"
rf_model_path = "saved_models/random_forest_v1.pkl"
gb_model_path = "saved_models/gradient_boosting_v1.pkl"

# --------------------------
# Example input — you can include only a subset of features
student_input = {
    "age": 23,
    "hours_per_week": 10,
    "country": "India",
    # Add other features as needed; missing features will be auto-filled
}

# --------------------------
# Function to test a model
def test_model(model_path, model_name):
    print(f"\nTesting {model_name}...")
    model = InferenceModel(model_path=model_path)
    result = model.predict(student_input)
    print(f"Prediction result for {model_name}:", result)

# --------------------------
# Test all three models
test_model(baseline_model_path, "Baseline Logistic Regression")
test_model(rf_model_path, "Random Forest")
test_model(gb_model_path, "Gradient Boosting")