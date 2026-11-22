# Crop Yield Predictor (FastAPI)

## Structure
```
crop_app/
├── src/
│   ├── app.py      FastAPI app (endpoints + serves the webpage)
│   ├── ml.py       loads the model, prepares inputs, predicts
│   └── db.py       SQLite database (predictions + uploads)
├── static/index.html      single webpage
├── models/                crop_yield_model.joblib, scaler.joblib, model_columns.joblib
├── sample_data/sample_input.csv
├── tests/test_app.py
├── requirements.txt
└── crop_app.db            created automatically
```

## Endpoints
| Method | URL | Does |
|---|---|---|
| GET | `/` | webpage |
| GET | `/health` | status check |
| GET | `/options` | countries + crops for dropdowns |
| POST | `/predict` | one prediction, saved to DB |
| POST | `/upload-csv` | predict every CSV row, saved to DB |
| GET | `/history` | saved predictions |
| GET | `/uploads` | uploaded files |
| GET | `/docs` | API test page |

## Run (local)
```powershell
pip install -r requirements.txt
python -m uvicorn src.app:app --reload
```
Local URL: http://127.0.0.1:8000

## Test
```powershell
python -m pytest -v
```

## LAN URL (same Wi-Fi)
```powershell
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
ipconfig        # find IPv4, e.g. 192.168.1.25
```
LAN URL: http://192.168.1.25:8000

## Public URL
Option 1, Cloudflare tunnel (no account):
```powershell
cloudflared tunnel --url http://localhost:8000
```
Option 2, ngrok:
```powershell
ngrok http 8000
```

Option 3, Render (permanent): push to GitHub → render.com → New Web Service →
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`

(Render's free disk resets on restart, so the SQLite history is cleared then.)
