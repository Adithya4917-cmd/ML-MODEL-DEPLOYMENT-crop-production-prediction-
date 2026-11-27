import joblib

model = joblib.load("models/crop_yield_model.joblib")
joblib.dump(model, "models/crop_yield_model.joblib", compress=("xz", 9))
print("done")