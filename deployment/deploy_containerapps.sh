#!/usr/bin/env bash
set -e

# Load environment variables
set -a
source "$(dirname "$0")/.env"
set +a

echo "🔹 Logging in to Azure..."
az login --use-device-code

echo "🔹 Setting subscription..."
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

echo "🔹 Creating resource group (if needed)..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "🔹 Creating Azure Container Registry..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true

echo "🔹 Logging into ACR..."
az acr login --name "$ACR_NAME"

echo "🔹 Building Docker image inside Azure..."
az acr build \
  --registry "$ACR_NAME" \
  --image "$IMAGE_NAME:$IMAGE_TAG" \
  -f Dockerfile.azure .

echo "🔹 Creating Azure Container Apps environment..."
az containerapp env create \
  --name "$CONTAINERAPPS_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "🔹 Deploying Container App..."
az containerapp create \
  --name "$CONTAINER_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINERAPPS_ENV" \
  --image "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG" \
  --ingress external \
  --target-port 8000 \
  --registry-server "$ACR_NAME.azurecr.io"

echo "🔹 Fetching application URL..."
az containerapp show \
  --name "$CONTAINER_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv
