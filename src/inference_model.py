import pickle
import pandas as pd
from typing import Dict

class InferenceModel:
    def __init__(self, model_path: str):
        """
        Load a pre-trained ML model pipeline from a .pkl file.
        The pipeline should include preprocessing and classifier.
        """
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, input_data: Dict):
        """
        Accepts a single JSON-like dictionary input and returns prediction.
        Returns:
            dict: {"prediction": 0 or 1, "probability": float between 0-1}
        """
        # Convert input dict to single-row DataFrame
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = self.model.predict(df)
        probability = self.model.predict_proba(df)[:, 1]
        
        return {"prediction": int(prediction[0]), "probability": float(probability[0])}

# Example usage:
if __name__ == "__main__":
    # Choose which model to load
    model_paths = {
        "baseline": "saved_models/baseline_logreg.pkl",
        "random_forest": "saved_models/random_forest_v1.pkl",
        "gradient_boosting": "saved_models/gradient_boosting_v1.pkl"
    }

    model = InferenceModel(model_paths["baseline"])  # change key to switch models

    sample_input = {
        "age": 25,
        "hours_per_week": 10,
        "country": "USA",
        "num_logins_last_month": 15,
        "videos_watched_pct": 80,
        "assignments_submitted": 5,
        "discussion_posts": 3,
        "is_working_professional": 1,
        "preferred_device": "Laptop",
        "weight_kg": 65,
        "height_cm": 170
    }

    result = model.predict(sample_input)
    print("Prediction result:", result)