#!/usr/bin/env bash
set -e

# Load environment variables
set -a
source "$(dirname "$0")/.env"
set +a

echo "🔹 Building and pushing new image to ACR..."
az acr build \
  --registry "$ACR_NAME" \
  --image "$IMAGE_NAME:$IMAGE_TAG" \
  -f Dockerfile.azure .

echo "🔹 Updating running container app..."
az containerapp update \
  --name "$CONTAINER_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG"

echo "🔹 Done! Updated app URL:"
az containerapp show \
  --name "$CONTAINER_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv
