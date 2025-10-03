import joblib
import pandas as pd
import numpy as np

class InferenceModel:
    def __init__(self, model_path: str):
        """
        Initialize the inference model.
        model_path: path to the saved .joblib file
        """
        self.model = joblib.load(model_path)
        
        # Save feature names and types for missing-column handling
        # Extract feature names used in training from the ColumnTransformer
        self.feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []

        preprocessor = self.model.named_steps['preprocessor']
        for name, transformer, cols in preprocessor.transformers_:
            self.feature_names.extend(cols)
            if name == 'num':
                self.numeric_cols.extend(cols)
            elif name == 'cat':
                self.categorical_cols.extend(cols)

    def predict(self, input_data: dict):
        """
        Run inference on a single input.
        input_data: dictionary with any subset of features.
        Missing features are auto-filled with 0 (numeric) or 'missing' (categorical)
        """
        # Fill missing columns
        for col in self.feature_names:
            if col not in input_data:
                if col in self.numeric_cols:
                    input_data[col] = 0
                else:
                    input_data[col] = "missing"

        # Convert dict → DataFrame
        X = pd.DataFrame([input_data])

        # Predict class (0/1) and probability
        y_pred = self.model.predict(X)[0]
        y_proba = self.model.predict_proba(X)[0][1]

        return {
            "prediction": int(y_pred),
            "probability": float(y_proba)
        }