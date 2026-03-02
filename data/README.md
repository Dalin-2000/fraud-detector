# Data Directory

This folder contains datasets used for training and evaluating the fraud detection model.

## Structure

```
data/
├── raw/        ← Original, unmodified source data (GITIGNORED)
└── processed/  ← Cleaned / feature-engineered data (GITIGNORED)
```

## Dataset

**File**: `V1-nigerian-financial-transactions-and-fraud-detection-dataset.csv`
**Size**: ~968 MB
**Source**: Nigerian financial transaction fraud detection dataset

### Setup

The dataset is NOT committed to git (size exceeds GitHub's 100 MB limit).
To set up locally, copy the CSV file into `data/raw/`:

```bash
cp ~/Desktop/Fraud\ Detection/V1-nigerian-financial-transactions-and-fraud-detection-dataset.csv \
   data/raw/
```

## Notebook

Open `notebooks/Fraud_Detection.ipynb` to run the full training pipeline.
The notebook reads from `data/raw/` and writes the trained model to `backend/ml_models/fraud_model.pkl`.
