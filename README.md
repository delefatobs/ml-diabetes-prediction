# ml-diabetes-prediction

This repository contains my end-to-end implementation of the ML midterm project.  
I use the CDC Diabetes Health Indicators dataset to build a diabetes prediction model, package it as an API, containerize it with Docker, and optionally deploy it to Azure Container Apps.

My goal with this project was to keep things simple, reproducible, and close to the kind of workflow I use in real data engineering work.

---
## Problem statement

Type 2 diabetes is a major public health challenge, and early detection makes a significant difference.  
This project uses the **CDC Diabetes Health Indicators** dataset to predict whether a person is diabetic (`Diabetes_binary = 1`) based on lifestyle, health, and demographic factors.

Machine learning helps by providing:
- A quick estimate of a person’s diabetes risk
- An additional signal that can support early screening
- Insight into which features contribute most to risk

My focus here is not clinical precision, but demonstrating the full ML lifecycle: EDA, model training, tuning, packaging, and deployment.

---
## Dataset

Dataset: **CDC Diabetes Health Indicators (BRFSS 2015)**  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

To use the dataset:
1. Download the CSV from Kaggle.
2. Create a folder named `data/` in the repository.
3. Save the file as:

