# Fraud API – Docker Starter

This starter turns your trained model into a web API using **FastAPI** and **Docker**.

## What you need from Colab
Export two files from your notebook after training:
1. `fraud_model_rf.pkl` – your trained scikit-learn RandomForest model (via `joblib.dump`)
2. `scaler_params.json` – training stats and feature order, shaped like:
   ```json
   {
     "amount_mean": 0.0,
     "amount_scale": 1.0,
     "time_mean": 0.0,
     "time_scale": 1.0,
     "feature_order": ["Time", "V1", "V2", "...", "V28", "Amount"]
   }
   ```

## Files in this folder
- `app.py` – FastAPI app exposing `/predict` and `/health`
- `requirements.txt` – Python libs
- `Dockerfile` – container build instructions
- `.dockerignore` – excludes junk from build context
- `index.html` – simple HTML page you can open locally and point to the API
  (example only; edit to list your real features)

## Build & run locally
Place `fraud_model_rf.pkl` and `scaler_params.json` next to `app.py` **before** building.

```bash
docker build -t fraud-api:latest .
docker run --rm -p 8080:8080 fraud-api:latest
```

Visit http://localhost:8080/docs to try the API.

## Example request
```bash
curl -X POST http://localhost:8080/predict       -H "Content-Type: application/json"       -d '{"features":{"Time":10000,"V1":-1.23,"V2":0.45,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,"V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V14":0,"V15":0,"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,"V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,"V28":0,"Amount":34.56}}'
```

## Deploy options

### Google Cloud Run (serverless)
1. Install gcloud CLI and login:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud config set run/region asia-southeast1
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
   ```
2. Build & push with Cloud Build to Artifact Registry:
   ```bash
   gcloud artifacts repositories create fraud-repo --repository-format=docker --location=asia-southeast1
   gcloud builds submit --tag asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/fraud-repo/fraud-api:latest .
   ```
3. Deploy:
   ```bash
   gcloud run deploy fraud-api          --image=asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/fraud-repo/fraud-api:latest          --platform=managed          --allow-unauthenticated
   ```
   Copy the HTTPS URL printed by the command.

### Render / Railway (Dockerfile deploy)
- Push this folder to GitHub.
- Create a new Web Service, let it detect the Dockerfile.
- It will set `PORT` env. Our Dockerfile honors `${PORT:-8080}` automatically.
- After deploy finishes, use the public URL it gives you.

## Switch to Keras model
- Save your NN model as `fraud_model_keras.h5` and add `tensorflow` to `requirements.txt`.
- In `app.py`, replace joblib load with:
  ```python
  from tensorflow.keras.models import load_model
  model = load_model("fraud_model_keras.h5")
  # prob = float(model.predict(X)[0, 0])
  ```

## Troubleshooting
- **ModuleNotFoundError**: Add the missing package to `requirements.txt` and rebuild.
- **Different feature order**: Ensure `scaler_params.json["feature_order"]` matches exactly your training columns.
- **CORS blocked in browser**: In `app.py` CORS config, set your site in `allow_origins=[ "https://yourdomain" ]` and rebuild.
- **Port in use**: Change the left side of `-p HOSTPORT:8080` when running locally.
- **Large models**: Consider storing models on a mounted volume or downloading at startup from cloud storage.