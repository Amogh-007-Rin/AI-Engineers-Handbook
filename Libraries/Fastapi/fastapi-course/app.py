"""Small typed API whose security and idempotency contracts are testable."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=32)


class Prediction(BaseModel):
    id: int
    score: float
    model_version: str


@dataclass
class Repository:
    records: dict[str, Prediction] = field(default_factory=dict)
    next_id: int = 1
    lock: Lock = field(default_factory=Lock)

    def create(self, key: str, request: PredictionRequest) -> Prediction:
        with self.lock:
            if key in self.records:
                return self.records[key]
            score = sum(request.features) / len(request.features)
            result = Prediction(id=self.next_id, score=score, model_version="mean-v1")
            self.next_id += 1
            self.records[key] = result
            return result


repository = Repository()
app = FastAPI(title="Prediction contract", version="1.0.0")


async def authenticate(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != "test-secret":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions", response_model=Prediction, dependencies=[Depends(authenticate)])
async def create_prediction(
    request: PredictionRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> Prediction:
    if not idempotency_key or len(idempotency_key) > 128:
        raise HTTPException(status_code=400, detail="valid Idempotency-Key required")
    return repository.create(idempotency_key, request)
