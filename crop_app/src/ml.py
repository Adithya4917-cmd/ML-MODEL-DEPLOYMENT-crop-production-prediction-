# ML part: load the saved model and turn raw inputs into predictions
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# load ONCE when the server starts
model   = joblib.load(MODEL_DIR / "crop_yield_model.joblib")
scaler  = joblib.load(MODEL_DIR / "scaler.joblib")
columns = joblib.load(MODEL_DIR / "model_columns.joblib")

NUM_COLS = ["year", "log_area", "temp_change_annual_c", "temp_change_djf_c",
            "temp_change_mam_c", "temp_change_jja_c", "temp_change_son_c"]

# the columns a user (form or CSV) must give
INPUT_COLS = ["country", "crop", "year", "area_harvested_ha",
              "temp_change_annual_c", "temp_change_djf_c", "temp_change_mam_c",
              "temp_change_jja_c", "temp_change_son_c"]
n
# crops / countries the model was trained on (read from the saved column names)
KNOWN_CROPS     = sorted(c[len("crop_"):] for c in columns if c.startswith("crop_"))
KNOWN_COUNTRIES = sorted(c[len("country_"):] for c in columns if c.startswith("country_"))


def predict_df(df: pd.DataFrame) -> np.ndarray:
    """df has INPUT_COLS (one or many rows). Returns yield in t/ha for every row."""
    X = df[INPUT_COLS].copy()

    # same feature engineering as the notebook
    X["log_area"] = np.log1p(X["area_harvested_ha"])
    X = X.drop(columns=["area_harvested_ha"])

    # same one-hot encoding, same columns, same order as training
    X = pd.get_dummies(X, columns=["crop", "country"], dtype=int)
    X = X.reindex(columns=columns, fill_value=0)

    # same scaling as training
    X[NUM_COLS] = scaler.transform(X[NUM_COLS])

    # model predicts log(1 + yield) -> back to t/ha
    return np.expm1(model.predict(X))
