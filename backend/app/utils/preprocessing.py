import os
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "ml_models" / "fraud_model.pkl"

_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
        if path.exists():
            _bundle = joblib.load(path)
    return _bundle


def preprocess(transaction) -> np.ndarray:
    """
    Convert a TransactionInput into a 31-feature array matching the training
    pipeline from Fraud_Detection.ipynb (signal-injected model).

    Raw (19):
        Numerical : time_since_last_transaction, spending_deviation_score,
                    velocity_score, geo_anomaly_score, amount_ngn
        Time      : hour, day_of_week, day_of_month, month, is_weekend,
                    is_business_hours
        Categorical (label-encoded): transaction_type, merchant_category,
                    location, device_used, payment_channel, sender_persona
        Boolean   : bvn_linked, new_device_transaction

    Engineered (12):
        amount_zscore, location_fraud_rate, txn_type_fraud_rate,
        log_amount, velocity_x_geo, dev_x_log_amt, composite_risk,
        off_hours_new_device, high_risk_hour, geo_x_new_device,
        amount_x_velocity, risk_no_bvn
    """
    bundle = _load_bundle()
    encoders          = bundle.get("encoders", {})          if bundle else {}
    feature_names     = bundle.get("feature_names")         if bundle else None
    loc_fraud_map     = bundle.get("loc_fraud_map", {})     if bundle else {}
    loc_fraud_default = bundle.get("loc_fraud_default", 0.055) if bundle else 0.055
    tt_fraud_map      = bundle.get("tt_fraud_map", {})      if bundle else {}
    tt_fraud_default  = bundle.get("tt_fraud_default", 0.055)  if bundle else 0.055
    train_amt_mean    = bundle.get("train_amount_mean", 50000.0) if bundle else 50000.0
    train_amt_std     = bundle.get("train_amount_std", 40000.0)  if bundle else 40000.0

    # ── Time features ─────────────────────────────────────────────────────────
    ts = transaction.timestamp if transaction.timestamp else datetime.utcnow()
    hour              = ts.hour
    day_of_week       = ts.weekday()          # Mon=0 … Sun=6
    day_of_month      = ts.day
    month             = ts.month
    is_weekend        = 1 if day_of_week >= 5 else 0
    is_business_hours = 1 if 9 <= hour <= 17 else 0

    # ── Categorical encoding ────────────────────────────────────────────────
    def encode(col: str, value) -> int:
        if col in encoders:
            try:
                return int(encoders[col].transform([str(value)])[0])
            except Exception:
                return 0          # unseen label → 0
        return 0

    # ── Raw numeric inputs ──────────────────────────────────────────────────
    amt          = float(transaction.amount_ngn)
    vel          = float(transaction.velocity_score)
    geo          = float(transaction.geo_anomaly_score)
    dev          = float(transaction.spending_deviation_score)
    bvn          = int(transaction.bvn_linked)
    new_dev      = int(transaction.new_device_transaction)

    # ── Engineered features ─────────────────────────────────────────────────
    # amount z-score relative to training distribution (no per-sender history at inference)
    amount_zscore = float(
        np.clip((amt - train_amt_mean) / (train_amt_std + 1e-6), -10, 10)
    )

    # Lookup-based fraud rates (fallback to training median if unseen)
    location_fraud_rate = float(
        loc_fraud_map.get(str(transaction.location), loc_fraud_default)
    )
    txn_type_fraud_rate = float(
        tt_fraud_map.get(str(transaction.transaction_type), tt_fraud_default)
    )

    log_amount           = float(np.log1p(amt))
    velocity_x_geo       = vel * geo
    dev_x_log_amt        = dev * log_amount
    composite_risk       = vel / 10.0 + geo + dev / 5.0 + new_dev * 0.5
    off_hours_new_device = (1 - is_business_hours) * new_dev
    high_risk_hour       = 1 if hour in {0, 1, 2, 3, 22, 23} else 0
    geo_x_new_device     = geo * new_dev
    amount_x_velocity    = log_amount * vel
    risk_no_bvn          = composite_risk * (1 - bvn)

    # ── Assemble full feature dict ──────────────────────────────────────────
    row = {
        # raw
        "time_since_last_transaction": float(transaction.time_since_last_transaction),
        "spending_deviation_score":    dev,
        "velocity_score":              vel,
        "geo_anomaly_score":           geo,
        "amount_ngn":                  amt,
        "hour":                        hour,
        "day_of_week":                 day_of_week,
        "day_of_month":                day_of_month,
        "month":                       month,
        "is_weekend":                  is_weekend,
        "is_business_hours":           is_business_hours,
        "transaction_type":            encode("transaction_type",  transaction.transaction_type),
        "merchant_category":           encode("merchant_category", transaction.merchant_category),
        "location":                    encode("location",          transaction.location),
        "device_used":                 encode("device_used",       transaction.device_used),
        "payment_channel":             encode("payment_channel",   transaction.payment_channel),
        "sender_persona":              encode("sender_persona",    transaction.sender_persona),
        "bvn_linked":                  bvn,
        "new_device_transaction":      new_dev,
        # engineered
        "amount_zscore":               amount_zscore,
        "location_fraud_rate":         location_fraud_rate,
        "txn_type_fraud_rate":         txn_type_fraud_rate,
        "log_amount":                  log_amount,
        "velocity_x_geo":              velocity_x_geo,
        "dev_x_log_amt":               dev_x_log_amt,
        "composite_risk":              composite_risk,
        "off_hours_new_device":        off_hours_new_device,
        "high_risk_hour":              high_risk_hour,
        "geo_x_new_device":            geo_x_new_device,
        "amount_x_velocity":           amount_x_velocity,
        "risk_no_bvn":                 risk_no_bvn,
    }

    # Build array in exact training column order
    if feature_names:
        try:
            arr = np.array([row[f] for f in feature_names], dtype=float)
        except KeyError as e:
            missing = [f for f in feature_names if f not in row]
            raise ValueError(f"Feature(s) missing in preprocessing: {missing}") from e
    else:
        arr = np.array(list(row.values()), dtype=float)

    return arr.reshape(1, -1)
