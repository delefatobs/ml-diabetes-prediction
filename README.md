# 🩺 Diabetes Prediction — End-to-End Machine Learning Project

This repository contains my complete implementation of the ML Midterm Project, demonstrating a full end-to-end machine learning workflow. Using the **Diabetes Prediction Dataset** from Kaggle, I built a predictive model, explored the data, trained and tuned multiple algorithms, built an API using FastAPI, containerized it with Docker, and deployed it to Azure Container Apps.

The goal is to build a clean, reproducible ML project following modern data engineering and MLOps practices.

---

# Table of Contents

1. Problem Statement  
2. Dataset  
3. Exploratory Data Analysis (EDA)  
4. Model Training  
5. Environment & Dependencies  
6. Training the Model  
7. Running the FastAPI Service  
8. API Usage Examples  
9. Docker Containerization  
10. Deployment to Azure  
11. Deployment Evidence  
12. Project Structure  

---

# 1. Problem Statement

Type 2 diabetes is a significant global health challenge. Early detection can dramatically reduce health risks, improve outcomes, and support prevention efforts.

This project builds a machine learning model that predicts whether an individual has diabetes (`diabetes = 1`) based on:

- Age  
- Gender  
- BMI  
- Blood glucose level  
- HbA1c  
- Smoking history  
- Hypertension  
- Heart disease  

While this model is **not** intended for clinical decision-making, it provides a complete demonstration of:

- Data ingestion  
- Cleaning and preprocessing  
- Exploratory data analysis (EDA)  
- Model selection  
- Hyperparameter tuning  
- Packaging into a deployed prediction API  
- Docker-based containerization  
- Cloud deployment  

---

# 2. Dataset

Dataset: **Diabetes Prediction Dataset**  
Source: Kaggle  
Download link available on Kaggle.

## How to include the dataset in this project

1. Download the dataset from Kaggle  
2. Create a folder called:

```
data/
```

3. Save the CSV as:

```
data/diabetes.csv
```

This ensures full reproducibility of both the notebook and the training script.

---

# 3. Exploratory Data Analysis (EDA)

The notebook (`notebook.ipynb`) includes extensive EDA:

### Data cleaning
- Missing values check  
- Data type corrections  
- Duplicate removal  
- Normalization of categorical values  

### Feature exploration
- Distribution plots  
- Bar charts for categorical features  
- Correlation heatmap  
- Outlier detection  

### Target analysis
- Diabetes class imbalance  
- Stratified sampling decisions  

### Feature importance
- Random Forest feature importance  
- XGBoost feature importance  
- Logistic regression coefficients  

---

# 4. Model Training

Multiple ML models were trained and compared:

- Logistic Regression (baseline)  
- Random Forest  
- XGBoost (baseline)  
- XGBoost (tuned using grid search)  

Hyperparameter tuning included parameters like:

- Learning rate  
- Max depth  
- Number of estimators  

The final selected pipeline is saved to:

```
model.pkl
```

---

# 5. Environment & Dependencies

The project uses **uv**, but also includes a `requirements.txt`.

## Using uv (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# 6. Training the Model

The training logic is exported into a script: `train.py`.

### Run the training script

```bash
uv run python train.py
```

or:

```bash
python train.py
```

This will:

- Load `data/diabetes.csv`  
- Clean and preprocess the dataset  
- Train the ML pipeline  
- Evaluate performance  
- Save `model.pkl` to the project root  

---

# 7. Running the FastAPI Service

`predict.py` exposes a FastAPI web service for online inference.

### Start the API locally

```bash
uv run uvicorn predict:app --reload
```

### API Endpoints

- Health check: http://127.0.0.1:8000/health  
- Swagger UI: http://127.0.0.1:8000/docs  

---

# 8. API Usage Examples

## Example JSON Request

```json
{
  "gender": "female",
  "age": 45,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 27.3,
  "HbA1c_level": 6.2,
  "blood_glucose_level": 150
}
```

## cURL

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"gender":"female","age":45,"hypertension":0,"heart_disease":0,"smoking_history":"never","bmi":27.3,"HbA1c_level":6.2,"blood_glucose_level":150}'
```

## Python

```python
import requests

data = {
  "gender": "female",
  "age": 45,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 27.3,
  "HbA1c_level": 6.2,
  "blood_glucose_level": 150
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
```

---

# 9. Docker Containerization

A complete Dockerfile is included.

### Build the Docker image

```bash
docker build -t diabetes-api .
```

### Run the container

```bash
docker run -p 8000:8000 diabetes-api
```

Open Swagger docs at:

```
http://localhost:8000/docs
```

---

# 10. Deployment to Azure

Deployment uses Azure Container Apps.

## Step 1: Create Resource Group

```bash
az group create --name diabetes-rg --location westeurope
```

## Step 2: Create Azure Container Registry (ACR)

```bash
az acr create --resource-group diabetes-rg --name diabetesacr123 --sku Basic
```

## Step 3: Login to ACR

```bash
az acr login --name diabetesacr123
```

## Step 4: Tag & Push Image

```bash
docker tag diabetes-api diabetesacr123.azurecr.io/diabetes-api:v1
docker push diabetesacr123.azurecr.io/diabetes-api:v1
```

## Step 5: Deploy to Azure Container Apps

```bash
az containerapp create   --name diabetes-api-service   --resource-group diabetes-rg   --image diabetesacr123.azurecr.io/diabetes-api:v1   --environment <your-container-environment>   --target-port 8000   --ingress external
```

## Step 6: Access the API

```
https://<your-app-name>.azurecontainerapps.io/docs
```

---

# 11. Deployment Evidence

Screenshots, URL, or video proof is stored here:

```
deployment/
└── azure-proof.png
```

---

# 12. Project Structure

```
project/
├── README.md
├── data/
│   └── diabetes.csv
├── notebook.ipynb
├── train.py
├── predict.py
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── deployment/
    └── azure-proof.png
```

---

# 🎯 Summary

This project demonstrates the full ML lifecycle, including:

- Full problem description  
- Extensive EDA  
- Multiple models + hyperparameter tuning  
- Training exported to script  
- Reproducible environment  
- FastAPI deployment  
- Docker containerization  
- Cloud deployment  
- Deployment proof  



