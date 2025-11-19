from fastapi import FastAPI
from api.schema import DiabetesRequest, DiabetesResponse
from src.models.predict import DiabetesModel

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="A FastAPI service that serves a tuned XGBoost model for diabetes prediction.",
    version="1.0.0"
)

# Load model only once when the API starts
model = DiabetesModel()


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=DiabetesResponse)
def predict(data: DiabetesRequest):
    """
    Predict whether a patient has diabetes based on input features.
    Returns prediction (0/1) and probability.
    """
    record = data.dict()

    prediction, probability = model.predict(record)

    return DiabetesResponse(
        prediction=prediction,
        probability=probability
    )
