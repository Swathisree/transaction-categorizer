"""
Example client that calls the local API. Run `uvicorn api:app --port 8000` in one
terminal, then `python client_example.py` in another.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"

def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())

# 1) Health
print("GET /health:", get("/health"))

# 2) Single prediction
example = {
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
print("\nPOST /predict (McDonald's €22.18):")
print(json.dumps(post("/predict", example), indent=2))

# 3) Batch prediction
batch = {"transactions": [
    example,
    {**example, "merchant_name": "CARREFOURMARKET", "amount_value": -45.30},
    {**example, "merchant_name": "NETFLIX.COM",     "amount_value": -13.49,
                 "pfm_recurring_frequency": "monthly",
                 "merchant_pan_entry_method": "CARD_ON_FILE"},
    {**example, "merchant_name": "SNCF VOYAGEURS",  "amount_value": -89.00,
                 "merchant_pan_entry_method": "MANUAL"},
    {**example, "merchant_name": "CASH SERVICES",   "amount_value": -40.00,
                 "merchant_pan_entry_method": "CHIP"},
]}
print("\nPOST /predict/batch (5 transactions):")
resp = post("/predict/batch", batch)
for t, p in zip(batch["transactions"], resp["predictions"]):
    print(f"  {t['merchant_name']:>20s}  €{t['amount_value']:>8.2f}  →  "
          f"{p['predicted_minor_category']:>14s}  /  {p['predicted_major_category']:<22s}  "
          f"(conf {p['confidence']*100:5.1f}%)")
