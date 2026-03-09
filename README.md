# Fraud Detector

A full-stack fraud detection application using a machine learning model trained on Nigerian financial transaction data.

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, scikit-learn
- **Frontend**: Angular 17+, TypeScript
- **ML**: Trained model (`fraud_model.pkl`) served via REST API

## Architecture

```
Angular Frontend  →  POST /predict  →  FastAPI Backend  →  fraud_model.pkl
     :4200                                 :8000
```

## Project Structure

```
fraud-detector/
├── docker-compose.yml        # Orchestrates all services
├── Dockerfile.streamlit      # Streamlit standalone UI
├── data/                     # Dataset files (gitignored — too large)
├── notebooks/                # Jupyter notebooks for training and exploration
├── backend/                  # FastAPI application
│   └── Dockerfile
└── frontend/                 # Angular application
    ├── Dockerfile
    └── nginx.conf
```

## Prerequisites

### Option A — Docker (recommended)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Option B — Manual
- Python 3.11+
- Node.js 18+
- Angular CLI (`npm install -g @angular/cli`)
- Jupyter (`pip install jupyter`)

## Getting Started

> Make sure `backend/ml_models/fraud_model.pkl` exists before running either option below.

---

### Option A — Docker (recommended)

1. Install and open [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Wait for the Docker whale icon in the menu bar to stop animating
3. Run:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Angular frontend | http://localhost:4200 |
| FastAPI backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

To stop:

```bash
docker compose down
```

---

### Option B — Manual

### 1. Data Setup

Copy the dataset into the data folder (not committed to git):

```bash
cp ~/Desktop/Fraud\ Detection/V1-nigerian-financial-transactions-and-fraud-detection-dataset.csv \
   data/raw/
```

### 2. Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Place your trained model at `backend/ml_models/fraud_model.pkl`.

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Running the Notebook

```bash
cd notebooks
jupyter notebook Fraud_Detection.ipynb
```

After training, export the model:

```python
import pickle
with open("../backend/ml_models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

### 5. Running the Application (Manual)

```bash
# Terminal 1 — Backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Docs available at http://localhost:8000/docs

# Terminal 2 — Frontend
cd frontend
ng serve
# App available at http://localhost:4200
```

## Running Tests

```bash
cd backend
pytest tests/
```
