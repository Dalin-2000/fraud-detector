import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# ── Load model bundle ──────────────────────────────────────────
@st.cache_resource
def load_model():
    path = Path("backend/ml_models/fraud_model.pkl")
    return joblib.load(path)

bundle = load_model()
model           = bundle['model']
encoders        = bundle['encoders']
feature_names   = bundle['feature_names']
threshold       = bundle['threshold']
loc_fraud_map   = bundle['loc_fraud_map']
loc_fraud_def   = bundle['loc_fraud_default']
tt_fraud_map    = bundle['tt_fraud_map']
tt_fraud_def    = bundle['tt_fraud_default']
amt_mean        = bundle['train_amount_mean']
amt_std         = bundle['train_amount_std']

# ── Sidebar — app info ─────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Fraud Detector")
    st.markdown("Nigerian Financial Transactions")
    st.divider()
    st.markdown("**Model:** XGBoost (Tuned)")
    st.markdown(f"**Threshold:** `{threshold:.3f}`")
    st.markdown(f"**Features:** `{len(feature_names)}`")
    st.divider()
    st.caption("Fill in the transaction details on the right to get a fraud prediction.")

# ── Title ──────────────────────────────────────────────────────
st.title("🛡️ Transaction Fraud Risk Analyzer")
st.markdown("Fill in the transaction details below to check if it is fraudulent.")
st.divider()

# ── Layout: form (left) | result (right) ──────────────────────
form_col, result_col = st.columns([2, 1], gap="large")

with form_col:
    st.subheader("📋 Transaction Details")

    # ── Section 1: Core Transaction ───────────────────────────
    st.markdown("#### 💳 Core Transaction")
    c1, c2 = st.columns(2)

    with c1:
        transaction_type = st.selectbox(
            "Transaction Type",
            options=sorted(encoders['transaction_type'].classes_),
            help="Type of financial transaction"
        )
        amount_ngn = st.number_input(
            "Amount (NGN) ₦",
            min_value=100.0,
            max_value=10_000_000.0,
            value=15_000.0,
            step=500.0,
            format="%.2f"
        )
        payment_channel = st.selectbox(
            "Payment Channel",
            options=sorted(encoders['payment_channel'].classes_),
            help="Route used to make the payment"
        )

    with c2:
        merchant_category = st.selectbox(
            "Merchant Category",
            options=sorted(encoders['merchant_category'].classes_),
            help="Type of merchant/business"
        )
        location = st.selectbox(
            "Location (City)",
            options=sorted(encoders['location'].classes_),
            help="City where the transaction occurred"
        )
        device_used = st.selectbox(
            "Device Used",
            options=sorted(encoders['device_used'].classes_),
            help="Channel or device used"
        )

    st.divider()

    # ── Section 2: Sender Profile ─────────────────────────────
    st.markdown("#### 👤 Sender Profile")
    c3, c4 = st.columns(2)

    with c3:
        sender_persona = st.selectbox(
            "Sender Persona",
            options=sorted(encoders['sender_persona'].classes_),
            help="Behavioral profile of the sender"
        )
        bvn_linked = st.radio(
            "BVN Linked?",
            options=[True, False],
            format_func=lambda x: "✅ Yes — BVN Linked" if x else "❌ No — No BVN",
            horizontal=True,
            help="Whether the account has a Bank Verification Number"
        )

    with c4:
        new_device_transaction = st.radio(
            "New Device?",
            options=[False, True],
            format_func=lambda x: "⚠️ Yes — New Device" if x else "✅ No — Known Device",
            horizontal=True,
            help="Is this the first time this device is used?"
        )

    st.divider()

    # ── Section 3: Risk Signals ───────────────────────────────
    st.markdown("#### ⚠️ Risk Signals")

    st.markdown("**Minutes Since Last Transaction**")
    time_options = {
        "Just now (< 1 min)": 0.5,
        "< 30 minutes": 15.0,
        "1 hour": 60.0,
        "3 hours": 180.0,
        "Half day": 720.0,
        "1 day+": 1440.0,
    }
    time_choice = st.select_slider(
        "Time since last transaction",
        options=list(time_options.keys()),
        value="1 hour",
        label_visibility="collapsed"
    )
    time_since_last_transaction = time_options[time_choice]
    st.caption(f"Selected: **{time_choice}** → `{time_since_last_transaction} mins`")

    st.markdown("**Spending Deviation Score**")
    dev_options = {
        "Normal (0)": 0.0,
        "Moderate (2)": 2.0,
        "Unusual (5)": 5.0,
        "Very unusual (8)": 8.0,
        "Extreme (10)": 10.0,
    }
    dev_choice = st.select_slider(
        "Spending deviation",
        options=list(dev_options.keys()),
        value="Normal (0)",
        label_visibility="collapsed"
    )
    spending_deviation_score = dev_options[dev_choice]
    st.caption(f"Selected: **{dev_choice}** → `{spending_deviation_score}`")

    st.markdown("**Velocity Score** *(how many rapid transactions)*")
    vel_options = {
        "None (0)": 0,
        "Low (1–3)": 2,
        "Medium (4–9)": 6,
        "High (10–15)": 12,
        "Very high (16+)": 18,
    }
    vel_choice = st.select_slider(
        "Velocity score",
        options=list(vel_options.keys()),
        value="None (0)",
        label_visibility="collapsed"
    )
    velocity_score = vel_options[vel_choice]
    st.caption(f"Selected: **{vel_choice}** → `{velocity_score}`")

    st.markdown("**Geo Anomaly Score** *(location unusualness)*")
    geo_options = {
        "Normal location (0.0)": 0.0,
        "Moderate anomaly (0.3)": 0.3,
        "Unusual location (0.7)": 0.7,
        "Impossible travel (1.0)": 1.0,
    }
    geo_choice = st.select_slider(
        "Geo anomaly score",
        options=list(geo_options.keys()),
        value="Normal location (0.0)",
        label_visibility="collapsed"
    )
    geo_anomaly_score = geo_options[geo_choice]
    st.caption(f"Selected: **{geo_choice}** → `{geo_anomaly_score}`")

    st.divider()

    # ── Submit button ─────────────────────────────────────────
    submitted = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )

# ── Result panel (right column) ───────────────────────────────
with result_col:
    st.subheader("📊 Risk Result")

    if not submitted:
        st.info("👈 Fill in the form and click **Analyze Transaction** to see the result.")

    else:
        # ── Feature engineering (mirrors notebook) ────────────
        now_hour        = 14          # default business hour for demo
        day_of_week     = 2
        day_of_month    = 15
        month           = 6
        is_weekend      = int(day_of_week >= 5)
        is_business_hrs = int(9 <= now_hour <= 17)

        log_amount      = float(np.log1p(amount_ngn))
        amount_zscore   = float(np.clip(
            (amount_ngn - amt_mean) / (amt_std + 1e-6), -10, 10
        ))

        loc_fraud_rate  = float(loc_fraud_map.get(location, loc_fraud_def))
        tt_fraud_rate   = float(tt_fraud_map.get(transaction_type, tt_fraud_def))

        velocity_x_geo      = float(velocity_score) * float(geo_anomaly_score)
        dev_x_log_amt       = float(spending_deviation_score) * log_amount
        composite_risk      = (
            float(velocity_score) / 10.0
            + float(geo_anomaly_score)
            + float(spending_deviation_score) / 5.0
            + float(new_device_transaction) * 0.5
        )
        off_hours_new_dev   = (1.0 - float(is_business_hrs)) * float(new_device_transaction)
        high_risk_hour      = int(now_hour in [0,1,2,3,22,23])
        geo_x_new_device    = float(geo_anomaly_score) * float(new_device_transaction)
        amount_x_velocity   = log_amount * float(velocity_score)
        risk_no_bvn         = composite_risk * (1.0 - float(bvn_linked))

        # ── Encode categoricals ───────────────────────────────
        def encode(col, val):
            le = encoders[col]
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            return int(mapping.get(str(val), -1))

        row = {
            'transaction_type':            encode('transaction_type', transaction_type),
            'merchant_category':           encode('merchant_category', merchant_category),
            'location':                    encode('location', location),
            'device_used':                 encode('device_used', device_used),
            'time_since_last_transaction': float(time_since_last_transaction),
            'spending_deviation_score':    float(spending_deviation_score),
            'velocity_score':              int(velocity_score),
            'geo_anomaly_score':           float(geo_anomaly_score),
            'payment_channel':             encode('payment_channel', payment_channel),
            'amount_ngn':                  float(amount_ngn),
            'bvn_linked':                  int(bvn_linked),
            'new_device_transaction':      int(new_device_transaction),
            'sender_persona':              encode('sender_persona', sender_persona),
            'hour':                        now_hour,
            'day_of_week':                 day_of_week,
            'day_of_month':                day_of_month,
            'month':                       month,
            'is_weekend':                  is_weekend,
            'is_business_hours':           is_business_hrs,
            'amount_zscore':               amount_zscore,
            'location_fraud_rate':         loc_fraud_rate,
            'txn_type_fraud_rate':         tt_fraud_rate,
            'log_amount':                  log_amount,
            'velocity_x_geo':              velocity_x_geo,
            'dev_x_log_amt':               dev_x_log_amt,
            'composite_risk':              composite_risk,
            'off_hours_new_device':        off_hours_new_dev,
            'high_risk_hour':              high_risk_hour,
            'geo_x_new_device':            geo_x_new_device,
            'amount_x_velocity':           amount_x_velocity,
            'risk_no_bvn':                 risk_no_bvn,
        }

        X = pd.DataFrame([row])[feature_names]
        prob  = float(model.predict_proba(X)[0][1])
        label = prob >= threshold

        # ── Result display ────────────────────────────────────
        if label:
            st.error("🚨 FRAUDULENT TRANSACTION")
            risk_color = "#ef4444"
            risk_label = "HIGH RISK"
        else:
            st.success("✅ LEGITIMATE TRANSACTION")
            risk_color = "#22c55e"
            risk_label = "LOW RISK"

        # Score gauge
        st.markdown(f"""
        <div style='text-align:center; padding: 1.5rem;
                    background: #f8fafc; border-radius: 12px;
                    border: 2px solid {risk_color}; margin-bottom: 1rem;'>
            <div style='font-size: 3rem; font-weight: 800; color: {risk_color};'>
                {prob*100:.1f}%
            </div>
            <div style='font-size: 1rem; color: #64748b; margin-top: 0.25rem;'>
                Fraud Probability
            </div>
            <div style='font-size: 0.85rem; font-weight: 700;
                        color: {risk_color}; margin-top: 0.5rem;
                        letter-spacing: 0.1em;'>
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        st.progress(prob, text=f"Risk Score: {prob*100:.1f}%")

        st.divider()

        # Risk breakdown
        st.markdown("**🔍 Risk Breakdown**")
        factors = {
            "Velocity Score":        min(velocity_score / 18.0, 1.0),
            "Geo Anomaly":           geo_anomaly_score,
            "Spending Deviation":    min(spending_deviation_score / 10.0, 1.0),
            "New Device":            float(new_device_transaction),
            "No BVN":                float(not bvn_linked),
            "Composite Risk":        min(composite_risk / 5.0, 1.0),
        }
        for factor, score in factors.items():
            color = "🔴" if score >= 0.7 else "🟡" if score >= 0.3 else "🟢"
            st.markdown(f"{color} **{factor}**")
            st.progress(float(score))

        st.divider()

        # Transaction summary
        st.markdown("**📄 Transaction Summary**")
        st.markdown(f"- **Amount:** ₦{amount_ngn:,.2f}")
        st.markdown(f"- **Type:** {transaction_type}")
        st.markdown(f"- **Channel:** {payment_channel}")
        st.markdown(f"- **Location:** {location}")
        st.markdown(f"- **Device:** {device_used}")
        st.markdown(f"- **Threshold used:** `{threshold:.3f}`")