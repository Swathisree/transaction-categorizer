"""
FastAPI server that exposes the tournament-champion Random Forest for
transaction categorization.

Endpoints:
    GET  /                      landing / quick usage
    GET  /health                liveness check
    GET  /categories            list minor + major categories and the lookup
    POST /predict               predict a single transaction
    POST /predict/batch         predict a list of transactions

Run (from this folder):

    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the auto-generated Swagger UI.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
META_PATH  = HERE / "model_meta.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transaction-api")

log.info("Loading model from %s ...", MODEL_PATH)
MODEL = joblib.load(MODEL_PATH)
META  = json.loads(META_PATH.read_text())
MINOR_TO_MAJOR = META["minor_to_major"]
MINOR_CLASSES  = META["minor_classes"]
MAJOR_CLASSES  = META["major_classes"]
log.info("Model loaded. %d minor classes, %d major classes.", len(MINOR_CLASSES), len(MAJOR_CLASSES))

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    """A single transaction to be categorized."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "amount_value": -22.18,
            "fees": 0.0,
            "created_at": "2026-04-28T11:45:21Z",
            "card_name": "Carte sans contact basique",
            "digitalization_type": "apple_pay",
            "merchant_pan_entry_method": "CHIP_CONTACTLESS",
            "status": "completed",
            "pfm_recurring_frequency": None,
            "merchant_name": "MAC DONALD S",
        }
    })

    amount_value: float = Field(..., description="Transaction amount in EUR (negative = debit).")
    fees: float = Field(0.0, description="Fee in EUR (0 for most rows).")
    created_at: datetime = Field(..., description="ISO 8601 timestamp with timezone.")
    card_name: str
    digitalization_type: str
    merchant_pan_entry_method: str
    status: str = Field("completed", description="'completed' or 'refunded'.")
    pfm_recurring_frequency: Optional[str] = Field(
        None, description="'monthly' or null (null is treated as 'none')."
    )
    merchant_name: str


class BatchRequest(BaseModel):
    transactions: List[Transaction]


class Prediction(BaseModel):
    predicted_minor_category: str
    predicted_major_category: str
    confidence: float = Field(..., description="Top-class probability from the Random Forest.")
    top3_minor: List[dict] = Field(
        ..., description="Top-3 minor categories with probabilities.",
    )


class BatchResponse(BaseModel):
    predictions: List[Prediction]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_dataframe(txns: List[Transaction]) -> pd.DataFrame:
    """Match the exact feature engineering done at training time."""
    rows = []
    for t in txns:
        ts = t.created_at
        rows.append({
            "amount_value": t.amount_value,
            "fees": t.fees,
            "hour": ts.hour,
            "dow":  ts.weekday(),
            "card_name": t.card_name,
            "digitalization_type": t.digitalization_type,
            "merchant_pan_entry_method": t.merchant_pan_entry_method,
            "status": t.status,
            "pfm_recurring_frequency": t.pfm_recurring_frequency or "none",
            "merchant_name": t.merchant_name,
        })
    return pd.DataFrame(rows)


def _predict(df: pd.DataFrame) -> List[Prediction]:
    probs = MODEL.predict_proba(df)
    classes = MODEL.classes_
    out: List[Prediction] = []
    for row in probs:
        order = row.argsort()[::-1]
        top_idx = order[0]
        pred_minor = str(classes[top_idx])
        pred_major = MINOR_TO_MAJOR[pred_minor]
        top3 = [
            {"minor_category": str(classes[i]), "probability": float(row[i])}
            for i in order[:3]
        ]
        out.append(Prediction(
            predicted_minor_category=pred_minor,
            predicted_major_category=pred_major,
            confidence=float(row[top_idx]),
            top3_minor=top3,
        ))
    return out


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Transaction categorizer",
    version="1.0.0",
    description=(
        "Classifies Lydia card transactions into a minor category (12 classes) "
        "and a major category (9 classes) using the tournament-winning Random Forest model."
    ),
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing():
    return """
    <html><head><title>Transaction categorizer</title>
    <style>
      body{font:15px/1.6 -apple-system,"SF Pro Text",Helvetica,Arial,sans-serif;
           max-width:720px;margin:40px auto;padding:0 24px;color:#0f172a;}
      code,pre{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:13px;}
      pre{padding:12px;overflow-x:auto;}
      a{color:#2563eb;text-decoration:none;}a:hover{text-decoration:underline;}
      .btn{display:inline-block;padding:8px 14px;background:#2563eb;color:white;
           border-radius:6px;margin:4px 6px 4px 0;}
    </style></head><body>
    <h1>Transaction categorizer API</h1>
    <p>Random Forest champion from the model tournament, serving predictions on both
    <code>minor_category</code> (12 classes) and <code>major_category</code> (9 classes, derived via lookup).</p>

    <a class="btn" href="/docs">Swagger UI</a>
    <a class="btn" href="/redoc">ReDoc</a>
    <a class="btn" href="/categories">Supported categories</a>

    <h2>Quick example</h2>
    <pre>curl -X POST http://127.0.0.1:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"amount_value": -22.18, "fees": 0.0,
          "created_at": "2026-04-28T11:45:21Z",
          "card_name": "Carte sans contact basique",
          "digitalization_type": "apple_pay",
          "merchant_pan_entry_method": "CHIP_CONTACTLESS",
          "status": "completed",
          "pfm_recurring_frequency": null,
          "merchant_name": "MAC DONALD S"}'</pre>

    <p>See <code>README.md</code> or <a href="/docs">/docs</a> for the full schema.</p>
    </body></html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": META["model_name"],
        "minor_classes": len(MINOR_CLASSES),
        "major_classes": len(MAJOR_CLASSES),
    }


@app.get("/categories")
def categories():
    return {
        "minor_classes": MINOR_CLASSES,
        "major_classes": MAJOR_CLASSES,
        "minor_to_major": MINOR_TO_MAJOR,
    }


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction):
    try:
        df = _to_dataframe([txn])
        return _predict(df)[0]
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not req.transactions:
        raise HTTPException(status_code=400, detail="transactions list is empty")
    try:
        df = _to_dataframe(req.transactions)
        return BatchResponse(predictions=_predict(df))
    except Exception as e:
        log.exception("predict_batch failed")
        raise HTTPException(status_code=400, detail=str(e))
