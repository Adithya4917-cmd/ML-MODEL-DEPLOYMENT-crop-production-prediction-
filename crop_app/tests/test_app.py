# Step 14: Testing.  Run from the crop_app folder:   python -m pytest -v
from pathlib import Path
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sample_input.csv"

GOOD = {
    "country": "India", "crop": "Rice", "year": 2024, "area_harvested_ha": 47000000,
    "temp_change_annual_c": 1.0, "temp_change_djf_c": 1.0, "temp_change_mam_c": 1.0,
    "temp_change_jja_c": 0.8, "temp_change_son_c": 1.1,
}


# ---------- application ----------
def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_webpage_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "Crop Yield Predictor" in r.text

def test_options_has_countries_and_crops():
    d = client.get("/options").json()
    assert len(d["countries"]) >= 1
    assert "Rice" in d["crops"]


# ---------- predictions ----------
def test_predict_works():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    y = r.json()["predicted_yield_t_ha"]
    assert 0.5 < y < 15                       # rice yield is a few t/ha

def test_sugar_cane_higher_than_lentils():
    cane   = client.post("/predict", json={**GOOD, "crop": "Sugar cane", "area_harvested_ha": 5000000}).json()
    lentil = client.post("/predict", json={**GOOD, "crop": "Lentils, dry", "area_harvested_ha": 1500000}).json()
    assert cane["predicted_yield_t_ha"] > 5 * lentil["predicted_yield_t_ha"]

def test_every_country_works():
    for country in client.get("/options").json()["countries"]:
        r = client.post("/predict", json={**GOOD, "country": country})
        assert r.status_code == 200, country


# ---------- bad inputs ----------
def test_unknown_crop():
    assert client.post("/predict", json={**GOOD, "crop": "Pizza"}).status_code == 400

def test_unknown_country():
    assert client.post("/predict", json={**GOOD, "country": "Mars"}).status_code == 400

def test_negative_area():
    assert client.post("/predict", json={**GOOD, "area_harvested_ha": -5}).status_code == 422

def test_missing_field():
    assert client.post("/predict", json={"country": "India"}).status_code == 422


# ---------- CSV upload + database ----------
def test_csv_upload_predicts_and_saves():
    before = len(client.get("/history?limit=10000").json())
    with open(SAMPLE, "rb") as f:
        r = client.post("/upload-csv", files={"file": ("sample_input.csv", f, "text/csv")})
    d = r.json()
    assert r.status_code == 200
    assert d["predicted_rows"] == d["total_rows"] > 0
    after = len(client.get("/history?limit=10000").json())
    assert after == before + d["predicted_rows"]          # rows were saved to the database

def test_csv_bad_rows_are_skipped():
    csv = ("country,crop,year,area_harvested_ha,temp_change_annual_c,temp_change_djf_c,"
           "temp_change_mam_c,temp_change_jja_c,temp_change_son_c\n"
           "India,Rice,2024,47000000,1,1,1,0.8,1.1\n"
           "India,Pizza,2024,100,1,1,1,1,1\n")
    d = client.post("/upload-csv", files={"file": ("t.csv", csv.encode(), "text/csv")}).json()
    assert d["predicted_rows"] == 1
    assert d["errors"][0]["problem"] == "unknown crop"

def test_csv_wrong_file_type():
    assert client.post("/upload-csv", files={"file": ("t.txt", b"hi", "text/plain")}).status_code == 400

def test_csv_missing_columns():
    assert client.post("/upload-csv", files={"file": ("t.csv", b"a,b\n1,2", "text/csv")}).status_code == 400

def test_uploads_listed():
    assert len(client.get("/uploads").json()) >= 1
