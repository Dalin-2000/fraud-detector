import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.hero h1 { margin:0; font-size:1.9rem; font-weight:800; letter-spacing:-0.02em; color:white !important; }
.hero p  { margin:.4rem 0 0; font-size:.95rem; color:#94a3b8 !important; }

.card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 0.75rem 1rem; margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
}
.card-title {
    font-size: .8rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #64748b; margin-bottom: .5rem;
}
.thin-div { border:none; border-top:1px solid #e2e8f0; margin:.9rem 0; }

.result-card  { border-radius:14px; padding:1.8rem; text-align:center; margin-bottom:1rem; }
.result-fraud { background:#fff1f2; border:2px solid #ef4444; }
.result-legit { background:#f0fdf4; border:2px solid #22c55e; }
.result-pct   { font-size:3.2rem; font-weight:900; line-height:1; margin-bottom:.3rem; }
.result-label { font-size:.8rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.result-sub   { font-size:.85rem; color:#64748b; margin-top:.2rem; }

.factor-row { display:flex; align-items:center; gap:.6rem; margin-bottom:.55rem; font-size:.88rem; }
.factor-label { flex:1; color:#374151; font-weight:500; }
.factor-bar-bg { flex:2; height:6px; background:#e5e7eb; border-radius:99px; overflow:hidden; }
.factor-bar-fill { height:100%; border-radius:99px; }

.summary-row { display:flex; justify-content:space-between; padding:.45rem 0; font-size:.875rem; border-bottom:1px solid #f1f5f9; }
.summary-row:last-child { border-bottom:none; }
.summary-key { color:#64748b; }
.summary-val { color:#0f172a; font-weight:600; }

[data-testid="stSidebar"] { background:#0f172a; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color:#1e293b !important; }
.sb-badge {
    display:inline-block; background:#1e3a5f; border:1px solid #334155;
    border-radius:6px; padding:.2rem .6rem; font-size:.8rem;
    font-family:monospace; color:#7dd3fc !important; margin-top:2px;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#1e40af,#0f172a) !important;
    border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-size:1rem !important;
    padding:.7rem 1rem !important; letter-spacing:.02em !important;
}
.stButton > button[kind="primary"]:hover { opacity:.88 !important; }
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div > div { border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

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

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## � How to Use")
    st.divider()
    st.markdown("""
**1. Core Transaction**  
Enter the amount, type, channel, merchant category, location, and device.

**2. Sender Profile**  
Select the sender persona and indicate whether the account has a BVN and whether the device is new.

**3. Risk Signals**  
Adjust the four sliders to reflect how unusual this transaction looks:
- ⏱ Time since the last transaction
- 📈 How much spending deviates from normal
- 🔁 Number of rapid back-to-back transactions
- 📍 How unusual the transaction location is

**4. Analyze**  
Click **Analyze Transaction** to get an instant fraud probability score and risk breakdown.
    """)

# ── Hero header ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ Transaction Fraud Risk Analyzer</h1>
    <p>Enter transaction details below to assess fraud likelihood using a trained XGBoost model.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: form (left) | result (right) ──────────────────────
form_col, result_col = st.columns([2, 1], gap="large")

with form_col:

    # ── Section 1: Core Transaction ───────────────────────────
    st.markdown('<div class="card"><div class="card-title">💳 Core Transaction</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        transaction_type = st.selectbox(
            "Transaction Type",
            options=sorted(encoders['transaction_type'].classes_),
            index=None, placeholder="Select type…",
            help="Type of financial transaction"
        )
        amount_ngn = st.number_input(
            "Amount (NGN) ₦",
            min_value=100.0,
            max_value=10_000_000.0,
            value=None,
            step=500.0,
            format="%.2f",
            placeholder="Enter amount"
        )
        payment_channel = st.selectbox(
            "Payment Channel",
            options=sorted(encoders['payment_channel'].classes_),
            index=None, placeholder="Select channel…",
            help="Route used to make the payment"
        )

    with c2:
        merchant_category = st.selectbox(
            "Merchant Category",
            options=sorted(encoders['merchant_category'].classes_),
            index=None, placeholder="Select category…",
            help="Type of merchant/business"
        )
        location = st.selectbox(
            "Location (City)",
            options=sorted(encoders['location'].classes_),
            index=None, placeholder="Select city…",
            help="City where the transaction occurred"
        )
        device_used = st.selectbox(
            "Device Used",
            options=sorted(encoders['device_used'].classes_),
            index=None, placeholder="Select device…",
            help="Channel or device used"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Sender Profile ─────────────────────────────
    st.markdown('<div class="card"><div class="card-title">👤 Sender Profile</div>', unsafe_allow_html=True)

    sender_persona = st.selectbox(
        "Sender Persona",
        options=sorted(encoders['sender_persona'].classes_),
        index=None, placeholder="Select persona…",
        help="Behavioral profile of the sender"
    )

    c3, c4 = st.columns(2)
    with c3:
        bvn_linked = st.radio(
            "BVN Linked?",
            options=[True, False],
            format_func=lambda x: "Yes — Verified" if x else "No — Unverified",
            index=None,
            horizontal=True,
            help="Whether the account has a Bank Verification Number"
        )
    with c4:
        new_device_transaction = st.radio(
            "New Device?",
            options=[False, True],
            format_func=lambda x: "Yes — First use" if x else "No — Known device",
            index=None,
            horizontal=True,
            help="Is this the first time this device is used?"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Risk Signals ───────────────────────────────
    st.markdown('<div class="card"><div class="card-title">⚠️ Risk Signals</div>', unsafe_allow_html=True)
    time_options = {
        "Just now (< 1 min)": 0.5,
        "< 30 minutes":       15.0,
        "1 hour":             60.0,
        "3 hours":            180.0,
        "Half day":           720.0,
        "1 day+":             1440.0,
    }
    time_choice = st.select_slider(
        "⏱  Minutes Since Last Transaction",
        options=list(time_options.keys()),
        value="1 hour",
    )
    time_since_last_transaction = time_options[time_choice]

    st.markdown('<hr class="thin-div">', unsafe_allow_html=True)

    dev_options = {
        "Normal (0)":        0.0,
        "Moderate (2)":      2.0,
        "Unusual (5)":       5.0,
        "Very unusual (8)":  8.0,
        "Extreme (10)":      10.0,
    }
    dev_choice = st.select_slider(
        "📈  Spending Deviation Score",
        options=list(dev_options.keys()),
        value="Normal (0)",
    )
    spending_deviation_score = dev_options[dev_choice]

    st.markdown('<hr class="thin-div">', unsafe_allow_html=True)

    vel_options = {
        "None (0)":        0,
        "Low (1–3)":       2,
        "Medium (4–9)":    6,
        "High (10–15)":    12,
        "Very high (16+)": 18,
    }
    vel_choice = st.select_slider(
        "🔁  Velocity Score  — rapid back-to-back transactions",
        options=list(vel_options.keys()),
        value="None (0)",
    )
    velocity_score = vel_options[vel_choice]

    st.markdown('<hr class="thin-div">', unsafe_allow_html=True)

    geo_options = {
        "Normal location (0.0)":   0.0,
        "Moderate anomaly (0.3)":  0.3,
        "Unusual location (0.7)":  0.7,
        "Impossible travel (1.0)": 1.0,
    }
    geo_choice = st.select_slider(
        "📍  Geo Anomaly Score  — location unusualness",
        options=list(geo_options.keys()),
        value="Normal location (0.0)",
    )
    geo_anomaly_score = geo_options[geo_choice]

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Analyze button ────────────────────────────────────────
    submitted = st.button(
        "🔍  Analyze Transaction",
        type="primary",
        use_container_width=True,
    )

# ── Result panel ───────────────────────────────────────────────
with result_col:
    st.markdown("#### 📊 Risk Result")

    if not submitted:
        st.markdown("""
        <div style='
            background:#f8fafc; border:1px dashed #cbd5e1;
            border-radius:12px; padding:2rem 1.5rem;
            text-align:center; color:#94a3b8;
            font-size:.9rem; line-height:1.7;
        '>
            👈 Fill in the form<br>and click<br>
            <strong style="color:#475569">Analyze Transaction</strong><br>
            to see the prediction.
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Validate all fields filled ─────────────────────────
        missing = []
        if transaction_type is None: missing.append("Transaction Type")
        if amount_ngn is None:       missing.append("Amount")
        if payment_channel is None:  missing.append("Payment Channel")
        if merchant_category is None:missing.append("Merchant Category")
        if location is None:         missing.append("Location")
        if device_used is None:      missing.append("Device Used")
        if sender_persona is None:   missing.append("Sender Persona")
        if bvn_linked is None:       missing.append("BVN Linked")
        if new_device_transaction is None: missing.append("New Device")

        if missing:
            st.warning(f"Please fill in: **{', '.join(missing)}**")
            st.stop()

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

        # ── Verdict card ───────────────────────────────────────
        if label:
            risk_color  = "#ef4444"
            risk_label  = "HIGH RISK"
            card_cls    = "result-fraud"
            verdict_txt = "🚨 FRAUDULENT"
        else:
            risk_color  = "#22c55e"
            risk_label  = "LOW RISK"
            card_cls    = "result-legit"
            verdict_txt = "✅ LEGITIMATE"

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="result-pct" style="color:{risk_color};">{prob*100:.1f}%</div>
            <div class="result-sub">Fraud Probability</div>
            <div class="result-label" style="color:{risk_color}; margin-top:.6rem;">
                {verdict_txt} &nbsp;·&nbsp; {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(prob)

        # ── Risk factor breakdown ──────────────────────────────
        st.markdown("<br>**Risk Factor Breakdown**", unsafe_allow_html=True)

        def _bar(name, score):
            pct = min(score * 100, 100)
            if score >= 0.7:
                fill_color, dot = "#ef4444", "🔴"
            elif score >= 0.3:
                fill_color, dot = "#f59e0b", "🟡"
            else:
                fill_color, dot = "#22c55e", "🟢"
            st.markdown(f"""
            <div class="factor-row">
                <span>{dot}</span>
                <span class="factor-label">{name}</span>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill"
                         style="width:{pct:.0f}%;background:{fill_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        _bar("Velocity",           min(velocity_score / 18.0, 1.0))
        _bar("Geo Anomaly",        geo_anomaly_score)
        _bar("Spending Deviation", min(spending_deviation_score / 10.0, 1.0))
        _bar("New Device",         float(new_device_transaction))
        _bar("No BVN",             float(not bvn_linked))
        _bar("Composite Risk",     min(composite_risk / 5.0, 1.0))

        # ── Transaction summary ────────────────────────────────
        st.markdown("<br>**Transaction Summary**", unsafe_allow_html=True)
        summary = [
            ("Amount",    f"₦{amount_ngn:,.2f}"),
            ("Type",      transaction_type),
            ("Channel",   payment_channel),
            ("Location",  location),
            ("Device",    device_used),
            ("Threshold", f"{threshold:.3f}"),
        ]
        rows_html = "".join(
            f'<div class="summary-row">'
            f'<span class="summary-key">{k}</span>'
            f'<span class="summary-val">{v}</span>'
            f'</div>'
            for k, v in summary
        )
        st.markdown(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-radius:10px;padding:.8rem 1rem;">{rows_html}</div>',
            unsafe_allow_html=True
        )

        def _render_field(field, values, encoders):
            name  = field['name']
            label = field['label']
            ftype = field['type']
        
            if ftype == 'select':
                options = [''] + sorted(encoders[name].classes_) if name in encoders else ['']
                values[name] = st.selectbox(label, options=options,
                                            format_func=lambda x: f"Select {label}…" if x == '' else x,
                                            index=0)
        
            elif ftype == 'select_mapped':
                # Display with icons, store original value
                display_options = [''] + list(field['options'].keys())
                raw_options     = [''] + list(field['options'].values())
                choice = st.selectbox(
                    label,
                    options=display_options,
                    format_func=lambda x: f"Select {label}…" if x == '' else x,
                    index=0
                )
                values[name] = field['options'].get(choice, '')
        
            elif ftype == 'number':
                values[name] = st.number_input(
                    label,
                    min_value=float(field.get('min', 0)),
                    max_value=float(field.get('max', 1_000_000)),
                    value=None,
                    step=500.0,
                    format="%.2f",
                    placeholder="Enter amount…"
                )
        
            elif ftype == 'toggle':
                values[name] = st.radio(
                    label,
                    options=['', True, False],
                    format_func=lambda x: "Select…" if x == '' else ("✅ Yes" if x is True else "❌ No"),
                    horizontal=True,
                    index=0
                )
        
            elif ftype == 'pills':
                options = list(field['options'].keys())
                default = field.get('default', options[0])
                choice  = st.select_slider(label, options=options, value=default)
                values[name] = field['options'][choice]
