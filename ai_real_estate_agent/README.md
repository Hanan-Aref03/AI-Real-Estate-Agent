# AI Real Estate Agent

FastAPI and Streamlit app for house-price prediction using a compact Ames Housing feature set, a scikit-learn model, and a two-stage LLM workflow.

## What Is Included

- FastAPI backend with `/extract`, `/predict`, and `/health`
- Streamlit frontend
- Local training script for generating model artifacts
- Docker setup
- Railway deployment config

## Project Layout

```text
ai_real_estate_agent/
├── app/
├── models/
├── training/
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   └── train_model.py
├── ui/
├── Dockerfile
├── railway.json
└── requirements.txt
```

## Local Setup

```powershell
cd c:\Users\USER\Desktop\AI-Real-Estate-Agent\ai_real_estate_agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_STAGE1_MODEL=gpt-4.1-mini
OPENAI_STAGE2_MODEL=gpt-4.1-mini
```

If `OPENAI_API_KEY` is missing, the app still runs with the built-in mock LLM fallback.

## Training The ML Model

The Kaggle files are expected at:

- `training/data/train.csv`
- `training/data/test.csv`

Train and generate artifacts:

```powershell
cd c:\Users\USER\Desktop\AI-Real-Estate-Agent\ai_real_estate_agent
.venv\Scripts\activate
python training\train_model.py
```

This writes:

- `models/best_model.pkl`
- `models/feature_names.pkl`
- `models/train_stats.json`

## Run The Backend

```powershell
cd c:\Users\USER\Desktop\AI-Real-Estate-Agent\ai_real_estate_agent
.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful URLs:

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Run The Streamlit UI

In another terminal:

```powershell
cd c:\Users\USER\Desktop\AI-Real-Estate-Agent\ai_real_estate_agent
.venv\Scripts\activate
streamlit run ui\streamlit_app.py
```

UI URL:

- `http://localhost:8501`

## Quick API Tests

Health check:

```powershell
curl http://localhost:8000/health
```

Extraction only:

```powershell
curl -X POST http://localhost:8000/extract `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"A house built in 2003 with lot area 9600 sq ft, living area 1800 sq ft, first floor 1200 sq ft, garage area 500 sq ft, and total basement 950 sq ft.\",\"user_filled_features\":null}"
```

Prediction:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"A house built in 2003 with lot area 9600 sq ft, living area 1800 sq ft, first floor 1200 sq ft, garage area 500 sq ft, and total basement 950 sq ft.\",\"user_filled_features\":{\"year_remod_add\":2005,\"mas_vnr_area\":0,\"bsmt_unf_sf\":100}}"
```

## Docker

Build:

```powershell
docker build -t ai-real-estate-agent .
```

Run:

```powershell
docker run --rm -p 8000:8000 `
  -e OPENAI_API_KEY=your_key_here `
  ai-real-estate-agent
```

If you want the UI separately, run Streamlit on your host machine while the backend container runs on port `8000`.

## Deploy To Railway

1. Push this repo to GitHub.
2. In Railway, choose `New Project` then `Deploy from GitHub repo`.
3. Select the `ai_real_estate_agent` project root as the service root if Railway asks.
4. Railway will use `railway.json` and the `Dockerfile`.
5. Add environment variables:
   - `OPENAI_API_KEY`
   - `OPENAI_STAGE1_MODEL` optional
   - `OPENAI_STAGE2_MODEL` optional
6. Make sure `models/best_model.pkl`, `models/feature_names.pkl`, and `models/train_stats.json` exist in the repo or are otherwise included in the deployed image.
7. Deploy and open the generated Railway URL.

Railway CLI flow:

```powershell
npm i -g @railway/cli
railway login
railway init
railway up
railway open
```

## Notes

- The backend blocks `/predict` until all required features are available.
- If `models/best_model.pkl` is missing, `/extract` still works but `/predict` returns a service error until the trained model is created.
- The local training script uses these backend-aligned features:
  - `lot_area`
  - `year_built`
  - `year_remod_add`
  - `mas_vnr_area`
  - `bsmt_unf_sf`
  - `total_bsmt_sf`
  - `first_flr_sf`
  - `garage_area`
  - `living_area`
