import streamlit as st
import joblib
import json
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
html, body, [class*="css"] { 
    font-family: "Inter", "Segoe UI", sans-serif;
    color: #0f172a;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.30), transparent 28%),
        radial-gradient(circle at top right, rgba(253, 224, 71, 0.22), transparent 24%),
        linear-gradient(180deg, #fdfefe 0%, #f3f8ff 52%, #eef6ff 100%);
}
[data-testid="stHeader"] { background: rgba(255, 255, 255, 0); }
[data-testid="stMainBlockContainer"] { color: #0f172a; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #fff7ed 0%, #eef7ff 55%, #f0f9ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
    box-shadow: 0 20px 45px rgba(148, 163, 184, 0.16);
}
.hero h1 { margin:0; font-size:1.9rem; font-weight:800; letter-spacing:-0.02em; color:#0f172a !important; }
.hero p  { margin:.4rem 0 0; font-size:.95rem; color:#475569 !important; }

.card {
    background: rgba(255, 255, 255, 0.88); border: 1px solid #dbeafe; border-radius: 12px;
    padding: 0.75rem 1rem; margin-bottom: 0.75rem;
    box-shadow: 0 12px 30px rgba(148, 163, 184, 0.10);
    backdrop-filter: blur(6px);
}
.card-title {
    font-size: .8rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #2563eb; margin-bottom: .5rem;
}
.thin-div { border:none; border-top:1px solid #dbeafe; margin:.35rem 0 .45rem; }

.risk-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.1rem;
}

.risk-desc {
    font-size: 0.78rem;
    line-height: 1.35;
    color: #64748b;
    margin-bottom: 0.3rem;
}

.result-card  { border-radius:14px; padding:1.8rem; text-align:center; margin-bottom:1rem; }
.result-fraud { background:#fff1f2; border:2px solid #ef4444; }
.result-legit { background:#f0fdf4; border:2px solid #22c55e; }
.result-pct   { font-size:3.2rem; font-weight:900; line-height:1; margin-bottom:.3rem; }
.result-label { font-size:.8rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.result-sub   { font-size:.85rem; color:#64748b; margin-top:.2rem; }

.factor-row { display:flex; align-items:center; gap:.6rem; margin-bottom:.55rem; font-size:.88rem; }
.factor-label { flex:1; color:#374151; font-weight:500; }
.factor-bar-bg { flex:2; height:6px; background:#dbeafe; border-radius:99px; overflow:hidden; }
.factor-bar-fill { height:100%; border-radius:99px; }

.summary-row { display:flex; justify-content:space-between; padding:.45rem 0; font-size:.875rem; border-bottom:1px solid #f1f5f9; }
.summary-row:last-child { border-bottom:none; }
.summary-key { color:#64748b; }
.summary-val { color:#0f172a; font-weight:600; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
    border-right: 1px solid #dbeafe;
}
[data-testid="stSidebar"] * { color:#0f172a !important; }
[data-testid="stSidebar"] hr { border-color:#dbeafe !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
label {
    color: #0f172a !important;
}

div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="base-input"] input {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 10px !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] * {
    color: #0f172a !important;
}

div[data-baseweb="popover"] [role="listbox"] {
    background: #ffffff !important;
    border: 1px solid #dbeafe !important;
    border-radius: 12px !important;
    box-shadow: 0 16px 36px rgba(148, 163, 184, 0.18) !important;
}

div[data-baseweb="popover"] [role="option"] {
    background: #ffffff !important;
    color: #0f172a !important;
}

div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}

div[data-baseweb="popover"] [role="option"]:hover {
    background: #f8fbff !important;
}

div[data-baseweb="select"] > div:focus-within,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus,
[data-baseweb="base-input"] input:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.20) !important;
}

/* Hide browser spin buttons so numeric fields look cleaner */
[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
[data-testid="stNumberInput"] input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

[data-testid="stNumberInput"] input[type=number] {
    -moz-appearance: textfield;
    appearance: textfield;
}

/* Keep compact select fields tidy */
[data-testid="stSelectbox"] {
    margin-bottom: 0.2rem !important;
}

[data-testid="stSelectbox"] label {
    margin-bottom: 0.2rem !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2563eb,#0ea5e9) !important;
    border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-size:1rem !important;
    padding:.7rem 1rem !important; letter-spacing:.02em !important;
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22) !important;
}
.stButton > button[kind="primary"]:hover { opacity:.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(Path("backend/ml_models/fraud_model.pkl"))

@st.cache_data
def load_form_config():
    with open("form_config.json") as f:
        return json.load(f)

bundle        = load_model()
form_config   = load_form_config()
model         = bundle['model']
encoders      = bundle['encoders']
feature_names = bundle['feature_names']
threshold     = bundle['threshold']
loc_fraud_map = bundle['loc_fraud_map']
loc_fraud_def = bundle['loc_fraud_default']
tt_fraud_map  = bundle['tt_fraud_map']
tt_fraud_def  = bundle['tt_fraud_default']
amt_mean      = bundle['train_amount_mean']
amt_std       = bundle['train_amount_std']

# ── Field renderer ─────────────────────────────────────────────
def _render_field(field, values, encoders):
    name  = field['name']
    label = field['label']
    ftype = field['type']

    if ftype == 'select':
        opts = [''] + sorted(encoders[name].classes_) if name in encoders else ['']
        values[name] = st.selectbox(
            label, options=opts,
            format_func=lambda x: f"Select {label}…" if x == '' else x,
            index=0
        )

    elif ftype == 'select_mapped':
        display_opts = [''] + list(field['options'].keys())
        choice = st.selectbox(
            label, options=display_opts,
            format_func=lambda x: f"Select {label}…" if x == '' else x,
            index=0
        )
        values[name] = field['options'].get(choice, '')

    elif ftype == 'number':
        if name == 'amount_ngn':
            values[name] = st.text_input(
                label,
                value="",
                placeholder="Enter amount..."
            )
        else:
            values[name] = st.number_input(
                label,
                min_value=float(field.get('min', 0)),
                max_value=float(field.get('max', 1_000_000)),
                value=None,
                step=500.0,
                format="%.2f",
                placeholder="Enter amount..."
            )

    elif ftype == 'toggle':
        true_label  = field.get('true_label',  '✅ Yes')
        false_label = field.get('false_label', '❌ No')
        values[name] = st.radio(
            label,
            options=[True, False],
            format_func=lambda x, tl=true_label, fl=false_label: tl if x is True else fl,
            horizontal=False,
            index=None,
        )

    elif ftype == 'pills':
        opts    = list(field['options'].keys())
        default = field.get('default', opts[0])
        display_label = field.get('short_label', label)
        description = field.get('description')
        if not description and '—' in label:
            description = label.split('—', 1)[1].strip()

        st.markdown(
            f'<div class="risk-label">{display_label}</div>',
            unsafe_allow_html=True,
        )
        if description:
            st.markdown(
                f'<div class="risk-desc">{description}</div>',
                unsafe_allow_html=True,
            )

        choice  = st.selectbox(
            display_label,
            options=opts,
            index=opts.index(default),
            help=label,
            label_visibility="collapsed",
        )
        values[name] = field['options'][choice]

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.divider()
    st.markdown("""
**1. Core Transaction**
Enter the amount, type, channel, merchant category, location, and device.

**2. Sender Profile**
Select the sender persona and indicate whether the account has a BVN and whether the device is new.

**3. Risk Signals**
Choose the four preset risk levels to reflect how unusual this transaction looks.

**4. Analyze**
Click **Analyze Transaction** to get an instant fraud probability score.
    """)

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ Transaction Fraud Detector</h1>
    <p>Enter transaction details below to assess fraud likelihood using a trained model.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────
form_col, result_col = st.columns([2, 1], gap="large")
values = {}

with form_col:
    for section in form_config['sections']:
        st.markdown(
            f'<div class="card"><div class="card-title">{section["title"]}</div>',
            unsafe_allow_html=True
        )
        fields = section['fields']
        i = 0
        while i < len(fields):
            field = fields[i]
            ftype = field['type']

            if ftype == 'pills':
                # Pair compact risk selectors side by side
                if i + 1 < len(fields) and fields[i + 1]['type'] == 'pills':
                    c1, c2 = st.columns(2)
                    with c1:
                        _render_field(fields[i], values, encoders)
                    with c2:
                        _render_field(fields[i + 1], values, encoders)
                    i += 2
                else:
                    _render_field(field, values, encoders)
                    i += 1

            elif ftype == 'toggle':
                # Pair toggles side by side
                if i + 1 < len(fields) and fields[i + 1]['type'] == 'toggle':
                    c1, c2 = st.columns(2)
                    with c1:
                        _render_field(fields[i], values, encoders)
                    with c2:
                        _render_field(fields[i + 1], values, encoders)
                    i += 2
                else:
                    _render_field(field, values, encoders)
                    i += 1

            else:
                # Pair selects/numbers side by side
                if i + 1 < len(fields) and fields[i + 1]['type'] not in ['pills', 'toggle']:
                    c1, c2 = st.columns(2)
                    with c1:
                        _render_field(fields[i], values, encoders)
                    with c2:
                        _render_field(fields[i + 1], values, encoders)
                    i += 2
                else:
                    _render_field(field, values, encoders)
                    i += 1

        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.button("🔍  Analyze Transaction", type="primary", use_container_width=True)

# ── Result panel ───────────────────────────────────────────────
with result_col:
    st.markdown("#### 📊 Risk Result")

    if not submitted:
        st.markdown("""
        <div style='background:#f8fafc; border:1px dashed #cbd5e1;
            border-radius:12px; padding:2rem 1.5rem;
            text-align:center; color:#94a3b8; font-size:.9rem; line-height:1.7;'>
            👈 Fill in the form<br>and click<br>
            <strong style="color:#475569">Analyze Transaction</strong><br>
            to see the prediction.
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Validate ───────────────────────────────────────────
        missing = []
        for section in form_config['sections']:
            for field in section['fields']:
                name = field['name']
                val  = values.get(name)
                if field['type'] not in ['pills'] and (val is None or val == ''):
                    missing.append(field['label'])

        if missing:
            st.warning(f"Please fill in: **{', '.join(missing)}**")
            st.stop()

        # ── Pull values ────────────────────────────────────────
        transaction_type         = values['transaction_type']
        amount_raw = str(values['amount_ngn']).replace(',', '').strip()
        try:
            amount_ngn = float(amount_raw)
        except ValueError:
            st.warning("Please enter a valid amount.")
            st.stop()

        if amount_ngn <= 0:
            st.warning("Amount must be greater than zero.")
            st.stop()
        payment_channel          = values['payment_channel']
        merchant_category        = values['merchant_category']
        location                 = values['location']
        device_used              = values['device_used']
        sender_persona           = values['sender_persona']
        bvn_linked               = bool(values['bvn_linked'])
        new_device_transaction   = bool(values['new_device_transaction'])
        time_since_last_transaction  = float(values['time_since_last_transaction'])
        spending_deviation_score     = float(values['spending_deviation_score'])
        velocity_score               = int(values['velocity_score'])
        geo_anomaly_score            = float(values['geo_anomaly_score'])

        # ── Feature engineering ────────────────────────────────
        now_hour        = 14
        day_of_week     = 2
        day_of_month    = 15
        month           = 6
        is_weekend      = int(day_of_week >= 5)
        is_business_hrs = int(9 <= now_hour <= 17)

        log_amount    = float(np.log1p(amount_ngn))
        amount_zscore = float(np.clip((amount_ngn - amt_mean) / (amt_std + 1e-6), -10, 10))

        loc_fraud_rate = float(loc_fraud_map.get(location, loc_fraud_def))
        tt_fraud_rate  = float(tt_fraud_map.get(transaction_type, tt_fraud_def))

        composite_risk    = (float(velocity_score)/10.0 + float(geo_anomaly_score)
                             + float(spending_deviation_score)/5.0
                             + float(new_device_transaction)*0.5)
        velocity_x_geo    = float(velocity_score) * float(geo_anomaly_score)
        dev_x_log_amt     = float(spending_deviation_score) * log_amount
        off_hours_new_dev = (1.0 - float(is_business_hrs)) * float(new_device_transaction)
        high_risk_hour    = int(now_hour in [0,1,2,3,22,23])
        geo_x_new_device  = float(geo_anomaly_score) * float(new_device_transaction)
        amount_x_velocity = log_amount * float(velocity_score)
        risk_no_bvn       = composite_risk * (1.0 - float(bvn_linked))

        def encode(col, val):
            le = encoders[col]
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            return int(mapping.get(str(val), -1))

        row = {
            'transaction_type':            encode('transaction_type', transaction_type),
            'merchant_category':           encode('merchant_category', merchant_category),
            'location':                    encode('location', location),
            'device_used':                 encode('device_used', device_used),
            'time_since_last_transaction': time_since_last_transaction,
            'spending_deviation_score':    float(spending_deviation_score),
            'velocity_score':              velocity_score,
            'geo_anomaly_score':           float(geo_anomaly_score),
            'payment_channel':             encode('payment_channel', payment_channel),
            'amount_ngn':                  amount_ngn,
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

        X     = pd.DataFrame([row])[feature_names]
        prob  = float(model.predict_proba(X)[0][1])
        label = prob >= threshold

        # ── Verdict ────────────────────────────────────────────
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

        # ── Risk breakdown ─────────────────────────────────────
        st.markdown("<br>**Risk Factor Breakdown**", unsafe_allow_html=True)

        def _bar(name, score):
            pct = min(score * 100, 100)
            fill_color = "#ef4444" if score >= 0.7 else "#f59e0b" if score >= 0.3 else "#22c55e"
            dot        = "🔴" if score >= 0.7 else "🟡" if score >= 0.3 else "🟢"
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

        # ── Summary ────────────────────────────────────────────
        st.markdown("<br>**Transaction Summary**", unsafe_allow_html=True)
        summary = [
            ("Amount",    f"₦{amount_ngn:,.2f}"),
            ("Type",      transaction_type),
            ("Channel",   payment_channel),
            ("Merchant",  merchant_category),
            ("Location",  location),
            ("Device",    device_used),
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