from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.predict import predict_next_day_rainfall, predict_for_date

app = FastAPI(
    title="Agro Weather Forecast API",
    description="Predicts next-day rainfall for agricultural planning in Nigeria",
    version="1.0.0"
)


class PredictRequest(BaseModel):
    location: str
    latitude: float
    longitude: float

class PredictResponse(BaseModel):
    location: str
    predicted_rainfall_mm: float
    prediction_date: str
    based_on_data_up_to: str
    model: str

class DatePredictRequest(BaseModel):
    date: str  # expects "YYYY-MM-DD"
    latitude: float = 6.5244
    longitude: float = 3.3792

class DatePredictResponse(BaseModel):
    prediction_date: str
    predicted_rainfall_mm: float
    based_on_data_up_to: str
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "Ridge (α=100)", "project": "agro-rainfall-forecaster"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """_summary_

    Args:
        request (PredictRequest): _description_

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    try:
        result = predict_next_day_rainfall(
            latitude=request.latitude,
            longitude=request.longitude
        )
        return PredictResponse(location=request.location, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/date", response_model=DatePredictResponse)
def predict_by_date(request: DatePredictRequest):
    """_summary_

    Args:
        request (DatePredictRequest): _description_

    Raises:
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    try:
        pd.to_datetime(request.date)  # validate format
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid date format. Use YYYY-MM-DD.")
    try:
        result = predict_for_date(
            target_date=request.date,
            latitude=request.latitude,
            longitude=request.longitude
        )
        return DatePredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))