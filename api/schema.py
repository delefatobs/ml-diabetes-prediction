from pydantic import BaseModel, Field


class DiabetesRequest(BaseModel):
    """
    Input schema for diabetes prediction.
    Matches the features used in training.
    """

    gender: str = Field(..., example="female")
    age: float = Field(..., ge=0, le=120, example=45)
    hypertension: int = Field(..., ge=0, le=1, example=0)
    heart_disease: int = Field(..., ge=0, le=1, example=0)
    smoking_history: str = Field(..., example="never")
    bmi: float = Field(..., ge=10, le=80, example=28.5)
    HbA1c_level: float = Field(..., ge=3, le=20, example=6.1)
    blood_glucose_level: float = Field(..., ge=50, le=400, example=150)


class DiabetesResponse(BaseModel):
    """
    Output schema for prediction response.
    """

    prediction: int = Field(..., example=1)
    probability: float = Field(..., example=0.87)
