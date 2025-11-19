# 🚀 Deploy Diabetes API to Azure Container Apps

This guide explains how to deploy the FastAPI diabetes prediction API (packaged in Docker) to **Azure Container Apps**.

---

# 1️⃣ Fill in Environment Variables

Before running commands, fill in these variables **with your real Azure values**:

```bash
RESOURCE_GROUP="diabetes-rg"
LOCATION="westeurope"

ACR_NAME="diabetesacr123"
ACR_LOGIN_SERVER="$ACR_NAME.azurecr.io"

IMAGE_NAME="diabetes-api"
IMAGE_TAG="v1"

CONTAINERAPPS_ENVIRONMENT="diabetes-env"
CONTAINERAPPS_NAME="diabetes-api-service"
```

---

# 2️⃣ Login to Azure

```bash
az login
```

Check active subscription:

```bash
az account show
```

Switch subscription (if needed):

```bash
az account set --subscription "YOUR-SUB-ID"
```

---

# 3️⃣ Create Resource Group

```bash
az group create   --name $RESOURCE_GROUP   --location $LOCATION
```

---

# 4️⃣ Create Azure Container Registry (ACR)

```bash
az acr create   --resource-group $RESOURCE_GROUP   --name $ACR_NAME   --sku Basic
```

---

# 5️⃣ Build & Tag Docker Image (Locally)

```bash
docker build -t $IMAGE_NAME .
docker tag $IMAGE_NAME $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
```

---

# 6️⃣ Login to ACR

```bash
az acr login --name $ACR_NAME
```

---

# 7️⃣ Push Docker Image to ACR

```bash
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
```

---

# 8️⃣ Create Azure Container Apps Environment

```bash
az containerapp env create   --name $CONTAINERAPPS_ENVIRONMENT   --resource-group $RESOURCE_GROUP   --location $LOCATION
```

---

# 9️⃣ Deploy the FastAPI Container to Azure Container Apps

```bash
az containerapp create   --name $CONTAINERAPPS_NAME   --resource-group $RESOURCE_GROUP   --environment $CONTAINERAPPS_ENVIRONMENT   --image $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG   --target-port 8000   --ingress external   --registry-server $ACR_LOGIN_SERVER   --query properties.configuration.ingress.fqdn   -o tsv
```

This command outputs your **public HTTPS URL**.

---

# 🔟 Test the Live API

Open Swagger UI:

```
https://<your-url>/docs
```

Health check:

```bash
curl https://<your-url>/health
```

Prediction API call:

```bash
curl -X POST https://<your-url>/predict   -H "Content-Type: application/json"   -d '{
    "gender": "female",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 27.3,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 150
  }'
```

---

# 1️⃣1️⃣ Update Container App with New Image Versions

Build and push new version:

```bash
docker build -t $IMAGE_NAME .
docker tag $IMAGE_NAME $ACR_LOGIN_SERVER/$IMAGE_NAME:v2
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:v2
```

Update the running container:

```bash
az containerapp update   --name $CONTAINERAPPS_NAME   --resource-group $RESOURCE_GROUP   --image $ACR_LOGIN_SERVER/$IMAGE_NAME:v2
```

---

# 1️⃣2️⃣ Cleanup (Optional)

```bash
az group delete   --name $RESOURCE_GROUP   --yes --no-wait
```

---

# ✅ Deployment Complete

Your FastAPI diabetes prediction API is now live on Azure Container Apps and accessible over HTTPS.
