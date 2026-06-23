from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from .model_service import (
    MODEL_ID,
    ModelServiceError,
    bundle_status,
    model_metadata,
    predict as predict_student,
)

app = FastAPI(title="ML Web Service API", version="0.1.0")

# This project is designed for local frontend/backend development and Docker
# Compose, where the frontend may be served from a different port than the API.
# Production deployments should replace `*` with their allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelSummary(BaseModel):
    """Frontend-facing model metadata returned by `GET /models`."""

    # The model metadata intentionally includes research-architecture details,
    # not just form fields. The frontend uses these values to show that the
    # served artifact is the binary terminal-outcome bundle with pseudo-labeling.
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    description: str | None = None
    phase: str | None = None
    problem_type: str | None = Field(default=None, alias="problemType")
    unresolved_label: str | None = Field(default=None, alias="unresolvedLabel")
    pseudo_labeling: dict[str, object] = Field(
        default_factory=dict, alias="pseudoLabeling"
    )
    features: list["FeatureSummary"] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)


class FeatureSummary(BaseModel):
    """One input column needed by the active trained model."""

    name: str
    kind: Literal["categorical", "numeric", "unknown"]
    required: bool = True


class PredictRequest(BaseModel):
    """Prediction request submitted by the frontend form."""

    model: str
    features: dict[str, str | int | float | bool | None]


class PredictResponse(BaseModel):
    """Prediction response returned to the UI."""

    # The frontend uses camelCase `modelVersion`, while Python code follows
    # snake_case. `populate_by_name=True` lets us construct this model from
    # either spelling.
    model_config = ConfigDict(populate_by_name=True)

    model: str
    prediction: str
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    model_version: str | None = Field(default=None, alias="modelVersion")


@app.get("/health")
def health() -> dict[str, object]:
    """Report API availability and whether the model bundle can be loaded."""

    status = bundle_status()
    return {
        "status": "ok" if status["loaded"] else "degraded",
        "model": status,
    }


@app.get("/models")
def get_models() -> list[ModelSummary]:
    """Return available model metadata so the frontend can build its form."""

    try:
        return [ModelSummary.model_validate(model_metadata())]
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Model bundle unavailable: {exc}"
        ) from exc


@app.post("/predict")
def predict(payload: PredictRequest) -> PredictResponse:
    """Run inference for one user-submitted student record."""

    # The API contract is already model-aware so additional artifacts can be
    # added later without changing the request shape.
    if payload.model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Unknown model: {payload.model}")

    try:
        return PredictResponse.model_validate(predict_student(payload.features))
    except ModelServiceError as exc:
        # Validation/domain problems are client-correctable, so they get 422.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Unexpected library/model failures are treated as temporary service
        # failures rather than bad user input.
        raise HTTPException(
            status_code=503, detail=f"Prediction failed: {exc}"
        ) from exc