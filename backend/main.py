from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import io

from backend.schemas import ApplicantInput
from backend.services.prediction_service import prediction_service


# ============================================
# DRIFTING ORACLE — FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Drifting Oracle",
    description="Credit-risk prediction and explainability API",
    version="0.1.0"
)


# ============================================
# CORS
# Allows the React frontend to communicate
# with the FastAPI backend.
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# ROOT
# ============================================

@app.get("/")
def root():
    return {
        "name": "Drifting Oracle",
        "status": "running",
        "version": "0.1.0"
    }


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": prediction_service.model is not None,
        "preprocessor_loaded": (
            prediction_service.preprocessor is not None
        ),
        "expected_features": len(
            prediction_service.expected_columns
        )
    }


# ============================================
# SINGLE APPLICANT PREDICTION
# ============================================

@app.post("/predict")
def predict(applicant: ApplicantInput):

    try:
        probability, _ = prediction_service.predict(
            applicant.model_dump()
        )

        # MVP risk bands.
        # These are NOT official bank approval rules.
        if probability < 0.10:
            risk_level = "LOW"

        elif probability < 0.20:
            risk_level = "MEDIUM"

        else:
            risk_level = "HIGH"

        return {
            "application_id": applicant.application_id,
            "name": applicant.name,
            "risk_probability": round(
                probability,
                4
            ),
            "risk_level": risk_level,
            "model_version": "baseline-v1"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================
# BATCH CSV PREDICTION
# ============================================

@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...)
):

    # Only CSV for MVP.
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:
        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {str(e)}"
        )

    results = []

    for index, row in df.iterrows():

        applicant = row.to_dict()

        # Use provided identifiers if available.
        application_id = applicant.get(
            "application_id",
            f"APP-{index + 1:05d}"
        )

        name = applicant.get(
            "name",
            f"Applicant {index + 1}"
        )

        # Keep metadata available for response.
        applicant["application_id"] = application_id
        applicant["name"] = name

        try:

            probability, _ = (
                prediction_service.predict(
                    applicant
                )
            )

            if probability < 0.10:
                risk_level = "LOW"

            elif probability < 0.20:
                risk_level = "MEDIUM"

            else:
                risk_level = "HIGH"

            results.append({
                "application_id": application_id,
                "name": name,
                "risk_probability": round(
                    probability,
                    4
                ),
                "risk_level": risk_level
            })

        except Exception as e:

            results.append({
                "application_id": application_id,
                "name": name,
                "risk_probability": None,
                "risk_level": "ERROR",
                "error": str(e)
            })

    return {
        "total_applications": len(df),
        "successful_predictions": sum(
            1
            for result in results
            if result["risk_level"] != "ERROR"
        ),
        "failed_predictions": sum(
            1
            for result in results
            if result["risk_level"] == "ERROR"
        ),
        "results": results
    }