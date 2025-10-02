import pandas as pd
import joblib

# Paths
DATA_PATH = "/Users/tamanna/Documents/ML project /Task 1/data/processed/online_course_completion_clean.csv"
PIPELINE_PATH = "/Users/tamanna/Documents/ML project /Task 1/models/pipeline.pkl"

# Load pipeline
pipeline = joblib.load(PIPELINE_PATH)

# Load data
df = pd.read_csv(DATA_PATH)

# Same features as training
categorical_cols = [col for col in df.select_dtypes(include='object').columns if col != 'course_completed']
numerical_cols = df.select_dtypes(include='number').columns.tolist()
if 'device_type' in categorical_cols:
    categorical_cols.remove('device_type')

X = df[categorical_cols + numerical_cols]

# Predictions
predictions = pipeline.predict(X)
df['predicted_course_completed'] = predictions
print(df[['predicted_course_completed']].head())