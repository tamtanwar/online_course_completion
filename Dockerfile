# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install exact versions of packages to match your local environment
RUN pip install fastapi==0.104.0 uvicorn==0.23.0 \
    numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1 joblib==1.3.2

# Copy all project files into the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Run FastAPI server
CMD ["python", "main.py"]