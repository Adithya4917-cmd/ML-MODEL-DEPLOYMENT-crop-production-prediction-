# FastAPI app: the API + serves the single webpage
# Run from the crop_app folder:   python -m uvicorn src.app:app --reload
from pathlib import Path
import io

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import db
from src.ml import predict_df, INPUT_COLS, KNOWN_CROPS, KNOWN_COUNTRIES

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Crop Yield Predictor")
db.init_db()                      # creates crop_app.db + tables if they don't exist


# ---------- input for one prediction ----------
class CropInput(BaseModel):
    country: str
    crop: str
    year: int = Field(ge=1961, le=2050)
    area_harvested_ha: float = Field(gt=0)
    temp_change_annual_c: float
    temp_change_djf_c: float
    temp_change_mam_c: float
    temp_change_jja_c: float
    temp_change_son_c: float


# ---------- the webpage ----------
@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# ---------- health check ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- dropdown lists ----------
@app.get("/options")
def options():
    return {"countries": KNOWN_COUNTRIES, "crops": KNOWN_CROPS}


# ---------- 1. single prediction (from the form) ----------
@app.post("/predict")
def predict(data: CropInput):
    if data.country not in KNOWN_COUNTRIES:
        raise HTTPException(400, f"Unknown country: {data.country}")
    if data.crop not in KNOWN_CROPS:
        raise HTTPException(400, f"Unknown crop: {data.crop}")

    row = data.model_dump()
    pred = float(predict_df(pd.DataFrame([row]))[0])
    row["predicted_yield_t_ha"] = round(pred, 3)
    row["estimated_production_t"] = round(pred * data.area_harvested_ha, 1)

    db.add_predictions([row], source="form")          # save to database
    return row


# ---------- 2. CSV upload: predict every row + save ----------
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read the CSV file")

    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise HTTPException(400, f"CSV is missing columns: {missing}")

    df = df[INPUT_COLS].copy()

    # check every row; bad rows are reported, not predicted
    num = INPUT_COLS[2:]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    problem = pd.Series("", index=df.index)
    problem[~df["country"].isin(KNOWN_COUNTRIES)] = "unknown country"
    problem[~df["crop"].isin(KNOWN_CROPS)] = "unknown crop"
    problem[df[num].isna().any(axis=1)] = "missing or non-numeric value"
    problem[df["area_harvested_ha"] <= 0] = "area must be > 0"

    good = df[problem == ""].copy()
    if len(good) > 0:
        good["predicted_yield_t_ha"] = predict_df(good).round(3)
        good["estimated_production_t"] = (good["predicted_yield_t_ha"] * good["area_harvested_ha"]).round(1)

    upload_id = db.add_upload(file.filename, len(df), len(good))
    db.add_predictions(good.to_dict("records"), source="csv", upload_id=upload_id)

    errors = [{"row": int(i) + 2, "problem": p} for i, p in problem[problem != ""].items()]  # +2 = CSV line number
    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "total_rows": len(df),
        "predicted_rows": len(good),
        "errors": errors,
        "results": good.to_dict("records"),
    }


# ---------- 3. history from the database ----------
@app.get("/history")
def history(limit: int = 50):
    return db.get_history(limit)


@app.get("/uploads")
def uploads():
    return db.get_uploads()
