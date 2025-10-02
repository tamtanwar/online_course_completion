import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Correct paths
DATA_PATH = "/Users/tamanna/Documents/ML project /Task 1/data/processed/online_course_completion_clean.csv"
MODEL_PATH = "/Users/tamanna/Documents/ML project /Task 1/models/logistic_model.pkl"
PIPELINE_PATH = "/Users/tamanna/Documents/ML project /Task 1/models/pipeline.pkl"

# Load data
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# Define features
categorical_cols = [col for col in df.select_dtypes(include='object').columns if col != 'course_completed']
numerical_cols = df.select_dtypes(include='number').columns.tolist()

# Handle missing optional 'device_type'
if 'device_type' in categorical_cols:
    categorical_cols.remove('device_type')

X = df[categorical_cols + numerical_cols]
y = df['course_completed']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=500))
])

# Train
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification)