import streamlit as st
import pandas as pd
from pathlib import Path

# ### PHASE 4: extra imports (safe)
import numpy as np
import altair as alt  # for nicer charts
import json, datetime, pathlib

from ac_advisor.predictor import Predictor
from ac_advisor.features import make_features
from ac_advisor.comfort import comfort_risk, effect_hint
from ac_advisor.coach import log_interaction
from ac_advisor.coach2 import nearest_context_advice, band_from_temp, speed_bucket
from ac_advisor.guardrails import defog_risk, comfort_guard

# === Model lineup + metrics (UI only for now) ===
MODEL_LINEUP = [
    "CatBoost (Point)",
    "CatBoost (Quantile)",
    "GRU (seq)",
    "Transformer (seq)",
    "Mamba (exp)",
]

MODEL_METRICS = {
    "CatBoost (Point)":    {"MAE": 4.21,  "RMSE": 7.03,  "R2": 0.9996, "notes": "Default"},
    "CatBoost (Quantile)": {"MAE": 4.21,  "RMSE": 7.03,  "R2": 0.9996, "notes": "P10–P90 uncertainty"},
    "GRU (seq)":           {"MAE": 12.71, "RMSE": 21.77, "R2": 0.9960, "notes": "Sequence (32-step)"},
    "Transformer (seq)":   {"MAE": 13.18, "RMSE": 19.87, "R2": 0.9966, "notes": "Sequence (32-step)"},
    "Mamba (exp)":         {"MAE": 13.62, "RMSE": 23.33, "R2": 0.9953, "notes": "Experimental"},
}

# Availability flags (we’ll flip to True after we export weights in STEP 2/3)
MODEL_AVAILABLE = {
    "CatBoost (Point)": True,
    "CatBoost (Quantile)": False,   # set True later if you wire quantile models
    "GRU (seq)": False,
    "Transformer (seq)": False,
    "Mamba (exp)": False,           # stays False until we truly have Mamba weights
}

# --- Auto-toggle availability based on files present in ./models ---
from pathlib import Path as _P

def _has(*names: str) -> bool:
    m = _P("models")
    return all((m / n).exists() for n in names)

# Flip availability for the models you’ve actually exported
MODEL_AVAILABLE.update({
    "GRU (seq)":            _has("gru.pt"),
    "Transformer (seq)":    _has("transformer.pt"),
    "Mamba (exp)":          _has("mamba.pt"),                 # stays False if you didn’t train Mamba
    "CatBoost (Quantile)":  _has("cb_q10.cbm", "cb_q90.cbm"), # only if you wired quantile models
})

st.set_page_config(page_title="A/C Energy Advisor", page_icon="❄️", layout="wide")

# =========================
# AC Advisor — Header: Logo Left, Text Center
# =========================
from pathlib import Path
import base64
import streamlit as st

def _img_as_data_uri(p: Path) -> str | None:
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        ext = p.suffix.lower().lstrip(".") or "png"
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return None

def render_header():
    assets = Path("assets")
    candidates = [assets / "ac_logo.png", assets / "ac-advisor-logo.png", assets / "logo.png"]
    logo_src = next((_img_as_data_uri(p) for p in candidates if p.exists()), None)

    st.markdown("""
    <style>
      .header-container {
          display: grid;
          grid-template-columns: 80px 1fr 80px; /* left, center, right */
          align-items: center;
          background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%);
          border-radius: 12px;
          padding: 0.8rem 1.2rem;
          color: white;
          box-shadow: 0 4px 10px rgba(0,0,0,0.18);
          position: relative;
      }
      .header-logo {
          width: 140px;
          height: 90px;
          border-radius: 12px;
          background-color: white;
          object-fit: contain;
          justify-self: start;
          box-shadow: 0 2px 6px rgba(0,0,0,0.25);
      }
      .header-text {
          text-align: center;  /* 👈 centers text block */
      }
      .header-title {
          font-size: 1.8rem;
          font-weight: 750;
          margin: 0;
          letter-spacing: 0.3px;
      }
      .header-sub {
          font-size: 1.05rem;
          font-weight: 400;
          margin: 0;
          opacity: 0.95;
      }
      .tagline {
          text-align: center;
          margin-top: 0.6rem;
          font-size: 1.0rem;
          color: white;
      }
    </style>
    """, unsafe_allow_html=True)

    # --- HTML layout ---
    if logo_src:
        header_html = f"""
        <div class="header-container">
            <img src="{logo_src}" class="header-logo" alt="Logo">
            <div class="header-text">
                <div class="header-title">❄️ AC Advisor</div>
                <div class="header-sub">Intelligent Energy Optimization for Automobile Air Conditioning Systems</div>
            </div>
            <div></div> <!-- right spacer -->
        </div>
        <div class='tagline'><b>Predict • Simulate • Save</b> — Data-driven climate control insights</div>
        """
    else:
        header_html = """
        <div class="header-container">
            <div></div>
            <div class="header-text">
                <div class="header-title">❄️ AC Advisor</div>
                <div class="header-sub">Intelligent Energy Optimization for Automobile Air Conditioning Systems</div>
            </div>
            <div></div>
        </div>
        <div class='tagline'><b>Predict • Simulate • Save</b> — Data-driven climate control insights</div>
        """

    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")

# Render banner
render_header()



# ---- Safe "apply" mechanism + toast BEFORE widgets are created ----
def _apply_pending_changes():
    # apply any pending widget updates from previous run
    pending = st.session_state.pop("__pending_apply__", None)
    if isinstance(pending, dict):
        for k, v in pending.items():
            st.session_state[k] = v
    # toast (if requested)
    note = st.session_state.pop("__notify__", None)
    if note:
        try:
            st.toast(note)
        except Exception:
            st.success(note)

_apply_pending_changes()

DATA_PATH = Path("data/EnergyPredictionDataset_ReadyForModel.csv")

@st.cache_resource
def load_predictor(version="v1"):
    return Predictor(version=version)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "ambient_temp" not in df.columns:
        df["ambient_temp"] = df.get("temperature", pd.Series([24]*len(df)))
    if "cabin_temp" not in df.columns:
        df["cabin_temp"] = df["ambient_temp"] - df.get("ac_power_proxy", pd.Series(0, index=df.index)).clip(0,5)*0.1
    return df

@st.cache_resource(show_spinner=False)
def _load_seq_model_cached(name: str):
    mdl = REGISTRY[name]
    mdl.load()
    return mdl


# --- HVAC band helper (Cooling / Heating / Mild) ---
def hvac_band(ambient_c: float):
    """
    ≤ 9 °C → Heating-dominant
    9–15 °C → Mild / mixed
    ≥ 15 °C → Cooling-dominant
    """
    if ambient_c <= 9:
        return ("Heating-dominant (≤9 °C)", "cold", "#1e3a8a")    # dark blue
    if ambient_c >= 15:
        return ("Cooling-dominant (≥15 °C)", "hot", "#7c2d12")    # dark warm
    return ("Mild / mixed (9–15 °C)", "mild", "#374151")          # dark gray

# Choose which model to use
# If user selects Quantile, force v2 (quantile-enabled).
if 'selected_model' in locals() and selected_model == "CatBoost (Quantile)":
    model_version = "v2"
else:
    model_version = st.segmented_control("Model", options=["v1", "v2"], default="v1")

pred = load_predictor(version=model_version)
df = load_data()


# --- Multi-model registry (auto from metadata) ---
from ac_advisor.model_registry import build_registry
import json
from pathlib import Path as _P

def _load_seq_config_from_meta():
    m = _P("models")
    for name in ["metadata_gru.json", "metadata_transformer.json", "metadata_mamba.json"]:
        p = m / name
        if p.exists():
            try:
                meta = json.loads(p.read_text())
                cols = meta.get("feature_cols")
                win  = int(meta.get("seq_len", 32))
                if isinstance(cols, list) and len(cols) > 0:
                    return cols, win
            except Exception:
                pass
    return ["speed","ambient_temp","cabin_temp","fan","recirc","setpoint","rpm","voltage_drop","bat_voltage"], 32

FEATURE_COLS_DL, SEQ_WINDOW = _load_seq_config_from_meta()
REGISTRY = build_registry(feature_cols=FEATURE_COLS_DL, window=SEQ_WINDOW)




st.sidebar.success(f"CatBoost model loaded ✅  ({model_version})")

with st.sidebar:
    st.markdown("### Model")
    selected_model = st.selectbox("Select model", MODEL_LINEUP, index=0)

    # Availability message (predictions still use CatBoost until we enable others)
    if not MODEL_AVAILABLE.get(selected_model, False):
        st.info("⚠️ This model isn’t available on this device yet. Predictions fall back to CatBoost.")

    # Micro KPIs
    mm = MODEL_METRICS.get(selected_model, {})
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (W)",  f"{mm.get('MAE','—')}")
    c2.metric("RMSE (W)", f"{mm.get('RMSE','—')}")
    c3.metric("R²",       f"{mm.get('R2','—')}")

# --- Sidebar Reset Button ---
with st.sidebar:
    if st.button("Reset to baseline"):
        st.session_state["__pending_apply__"] = {"sp_delta": 0, "fan_delta": 0, "recirc_on": False}
        st.session_state["__notify__"] = "Reset controls to baseline"
        st.rerun()

# --- st.title("Energy Prediction for Automobile A/C — What-If + AI Coach") ---
st.caption("Two-pass prediction: Baseline vs Simulation → Estimated saving (W) & comfort.")

# --- Proxies (physics/SHAP-informed) ---
proxy_cols = {
    "setpoint": "voltage_drop",  # warmer setpoint → lower compressor load (hot band)
    "fan":      "rpm",           # fan speed ↔ rpm
    "recirc":   "bat_voltage"    # recirc impacts electrical load
}
st.caption(
    f"Using proxy columns → Setpoint: `{proxy_cols['setpoint']}`, "
    f"Fan: `{proxy_cols['fan']}`, Recirc: `{proxy_cols['recirc']}`"
)

missing = [p for p in proxy_cols.values() if p not in pred.feature_cols]
if missing:
    st.warning(
        "This model does not include: " + ", ".join(missing) +
        ". Sliders may have no effect. Consider refreshing SHAP or retraining."
    )

# --- Controls ---
c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1.2,1.2,1.5])
with c1:
    row_idx = st.slider("Row index", 0, len(df)-1, min(50, len(df)-1), step=1, help="Select a sample data row")
st.markdown("#### Selected Data Row")
st.dataframe(df.iloc[[row_idx]].head(1), use_container_width=True)

with c2:
    setpoint_delta = st.slider("Setpoint Δ (°C)", -15, 15, 0, key="sp_delta")
with c3:
    fan_delta = st.slider("Fan level Δ", -3, 3, 0, help="Negative = lower fan, Positive = higher fan", key="fan_delta")
with c4:
    recirc_on = st.toggle("Recirculation ON", value=False, key="recirc_on")
with c5:
    sensitivity = st.slider("Effect sensitivity (×)", 0.5, 3.0, 1.2, key="sens")

vehicle_type = "ICE"  # ICE-only dataset

row = df.iloc[[row_idx]].copy()
ambient = float(row["ambient_temp"].iloc[0])
cabin   = float(row["cabin_temp"].iloc[0])
dT = round(cabin - ambient, 2)

# --- Band badge + info ---
mode_label, mode_key, mode_color = hvac_band(ambient)
st.markdown(
    f"""
    <div style="display:inline-block;padding:6px 12px;border-radius:999px;background:{mode_color};color:white;font-weight:600;margin-bottom:6px;">
        {mode_label}
    </div>
    """,
    unsafe_allow_html=True
)
hint = effect_hint(ambient, recirc_on)
# --- Comfort Band Info Banner ---
band_bg = {
    "cold": "#1e3a8a",   # blue for heating
    "mild": "#374151",   # gray for mild
    "hot":  "#7c2d12"    # brown-red for cooling
}.get(mode_key, "#334155")

st.markdown(
    f"""
    <div style="
        background:{band_bg};
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: white;
        font-size: 0.95rem;
        line-height: 1.55;
        box-shadow: 0 3px 8px rgba(0,0,0,0.25);
        margin-bottom: 0.5rem;
    ">
        <b>Ambient:</b> {ambient:.1f} °C &nbsp;
        <br><b>ΔT (cabin − ambient):</b> {dT:+.1f} °C  
        <br><b>Condition:</b> {hint}
        <br><b>Band rules:</b> Heating (≤ 9 °C)  •  Mild (9–15 °C)  •  Cooling (≥ 15 °C)
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("What do these bands mean?"):
    st.markdown(
        """
        - **Heating (≤ 9 °C)** → heater-dominant: warmer setpoint costs energy; fan adds a little; recirc benefit small.  
        - **Mild (9–15 °C)** → effects are modest and context-dependent.  
        - **Cooling (≥ 15 °C)** → warmer setpoint and recirc ON reduce A/C load; fan adds a little.
        """
    )

# --- Two-pass predictions (by selected_model) ---
use_seq = (
    selected_model in ("GRU (seq)", "Transformer (seq)", "Mamba (exp)")
    and MODEL_AVAILABLE.get(selected_model, False)
)

if use_seq:
    # 1) Build CatBoost-style features (keeps your physics/proxy logic consistent)
    X_now = make_features(
        row, setpoint_delta=0, fan_delta=0, recirc_on=False,
        sensitivity=1.0, predictor=pred, vehicle_type=vehicle_type, proxy_override=proxy_cols
    )
    X_sim = make_features(
        row, setpoint_delta=setpoint_delta, fan_delta=fan_delta, recirc_on=recirc_on,
        sensitivity=sensitivity, predictor=pred, vehicle_type=vehicle_type, proxy_override=proxy_cols
    )

    try:
        # 2) Build the SEQ_WINDOW history in FEATURE_COLS_DL order
        start = max(0, row_idx - SEQ_WINDOW + 1)
        hist = df.iloc[start:row_idx+1].reindex(columns=FEATURE_COLS_DL).astype(float)

        if len(hist) < SEQ_WINDOW:
            pad = np.repeat(hist.iloc[[0]].values, SEQ_WINDOW - len(hist), axis=0)
            seq_now = np.vstack([pad, hist.values])
        else:
            seq_now = hist.values[-SEQ_WINDOW:]

        # 3) Inject the EXACT proxy deltas CatBoost applied (so DL sees the same change)
        #    proxy_cols maps UI controls -> real columns (e.g., setpoint->voltage_drop, fan->rpm, recirc->bat_voltage)
        proxy_deltas = {}
        for _uifeat, proxy_col in proxy_cols.items():
            if proxy_col in X_now.columns and proxy_col in X_sim.columns:
                proxy_deltas[proxy_col] = float(X_sim[proxy_col].iloc[0] - X_now[proxy_col].iloc[0])

        seq_sim = seq_now.copy()
        for proxy_col, dval in proxy_deltas.items():
            if proxy_col in FEATURE_COLS_DL:
                j = FEATURE_COLS_DL.index(proxy_col)
                seq_sim[-1, j] = seq_sim[-1, j] + dval

        # 4) Predict with the selected sequence model
        mdl = _load_seq_model_cached(selected_model)
        mdl.load()
        x_now_vec = X_now.values[-1] if hasattr(X_now, "values") else np.array(X_now)
        x_sim_vec = X_sim.values[-1] if hasattr(X_sim, "values") else np.array(X_sim)
        pred_now = mdl.predict_one_row(X_now=x_now_vec, seq=seq_now)
        pred_sim = mdl.predict_one_row(X_now=x_sim_vec, seq=seq_sim)
        lo_now = hi_now = lo_sim = hi_sim = None  # no quantiles for DL models

    except Exception as e:
        # 5) Robust fallback so the UI never breaks
        st.warning(f"{selected_model} inference failed; falling back to CatBoost. Details: {e}")
        X_now = make_features(
            row, setpoint_delta=0, fan_delta=0, recirc_on=False,
            sensitivity=1.0, predictor=pred, vehicle_type=vehicle_type, proxy_override=proxy_cols
        )
        X_sim = make_features(
            row, setpoint_delta=setpoint_delta, fan_delta=fan_delta, recirc_on=recirc_on,
            sensitivity=sensitivity, predictor=pred, vehicle_type=vehicle_type, proxy_override=proxy_cols
        )
        pred_now_series, (lo_now, hi_now) = pred.predict(X_now)
        pred_sim_series, (lo_sim, hi_sim) = pred.predict(X_sim)
        pred_now = float(pred_now_series.iloc[0]); pred_sim = float(pred_sim_series.iloc[0])

else:
    # --- CatBoost path (original flow) ---
    X_now = make_features(
        row, setpoint_delta=0, fan_delta=0, recirc_on=False,
        sensitivity=1.0, predictor=pred, vehicle_type=vehicle_type, proxy_override=proxy_cols
    )
    X_sim = make_features(
        row, setpoint_delta=setpoint_delta, fan_delta=fan_delta, recirc_on=recirc_on,
        sensitivity=sensitivity, predictor=pred, vehicle_type=vehicle_type, proxy_override=proxy_cols
    )
    pred_now_series, (lo_now, hi_now) = pred.predict(X_now)
    pred_sim_series, (lo_sim, hi_sim) = pred.predict(X_sim)
    pred_now = float(pred_now_series.iloc[0]) 
    pred_sim = float(pred_sim_series.iloc[0])


saving_W = pred_now - pred_sim
saving_pct = (saving_W / max(1e-6, pred_now)) * 100.0

label, color = comfort_risk(ambient, cabin, setpoint_delta)

# --- Display metrics ---
lc, rc = st.columns([1.2,1])
with lc:
    st.metric("Baseline A/C power (W)", f"{pred_now:,.1f}")
    st.metric("Simulated A/C power (W)", f"{pred_sim:,.1f}")
with rc:
    st.metric("Estimated saving", f"{saving_W:,.1f} W", f"{saving_pct:+.1f}%")

# Prediction intervals (if quantile models exist)
if lo_now is not None and hi_now is not None:
    try:
        lo_b, hi_b = float(lo_now[0]), float(hi_now[0])
        st.caption(f"Baseline interval (P10–P90): {lo_b:.1f}–{hi_b:.1f} W")
    except Exception:
        pass
if lo_sim is not None and hi_sim is not None:
    try:
        lo_s, hi_s = float(lo_sim[0]), float(hi_sim[0])
        st.caption(f"Simulated interval (P10–P90): {lo_s:.1f}–{hi_s:.1f} W")
    except Exception:
        pass

# Comfort badge
comfort_colors = {"green": "#064e3b", "red": "#7f1d1d", "orange": "#78350f"}
comfort_bg = comfort_colors.get(color, "#1f2937")
st.markdown(
    f"""
    <div style="padding:10px 14px;border-radius:10px;background-color:{comfort_bg};color:white;font-weight:500;">
        Comfort: {label}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Vehicle scaling: **disabled (ICE-only dataset)**. Sensitivity scales the *visible* proxy change; the core model remains consistent.")

# --- Sidebar Export (placed AFTER predictions so all vars exist) ---
with st.sidebar:
    st.markdown("### Export Scenario")
    export_dir = Path("exports"); export_dir.mkdir(parents=True, exist_ok=True)
    export_md = f"""# A/C Scenario Snapshot

- Row index: **{row_idx}**
- Band: **{mode_label}**
- Ambient / Cabin: **{ambient:.1f} °C / {cabin:.1f} °C** (ΔT = {dT:+.1f} °C)

## Controls
- Setpoint Δ: **{setpoint_delta} °C**
- Fan Δ: **{fan_delta}**
- Recirculation: **{'ON' if recirc_on else 'OFF'}**
- Sensitivity: **{sensitivity:.2f}**

## Predictions
- Baseline A/C power: **{pred_now:.1f} W**
- Simulated A/C power: **{pred_sim:.1f} W**
- Estimated saving: **{saving_W:+.1f} W** ({saving_pct:+.1f}%)
"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = export_dir / f"scenario_{ts}.md"
    st.download_button("Download Markdown", data=export_md, file_name=fname.name, mime="text/markdown", use_container_width=True)
    if st.button("Save to exports/ folder"):
        with open(fname, "w") as f:
            f.write(export_md)
        st.success(f"Saved to: {fname.resolve()}")
        st.toast(f"Saved to: {fname.resolve()}")

# —— AI Coach v2 (ICE-only) ——
speed_val = float(row.get("speed", 0)) if "speed" in row.columns else None
coach = nearest_context_advice(ambient, speed_val)

with st.expander("Coach: best action for this context", expanded=True):
    st.caption(coach.get("explanation", ""))
    if coach.get("n", 0):
        st.caption(f"Samples in this context: **{coach.get('n', 0)}**")

    acts = coach.get("actions", [])
    MIN_GAIN_W = 1.0  # treat below as ≈0 W

    if not acts:
        st.info("Not enough history yet. Use the sliders a bit; this will fill in.")
    else:
        foggy = defog_risk(row.iloc[0]) if isinstance(row, pd.DataFrame) else False
        for a in acts[:3]:
            label = a["label"]
            est = a["est_W"]
            est_txt = "—" if (est is None or pd.isna(est)) else (f"{est:+.1f} W" if abs(est) >= MIN_GAIN_W else "≈0 W")

            # Comfort guard for setpoint suggestions
            block_for_comfort = False
            if "setpoint" in label.lower():
                proposed = 8 if "Increase" in label else (-8 if "Lower" in label else 0)
                block_for_comfort = comfort_guard(ambient, cabin, proposed)
            if block_for_comfort:
                st.write(f"• ~~{label}~~  → _blocked (comfort limit)_")
                st.session_state["__blocked_comfort__"] = True
                continue

            # Defog guard for Recirc
            if "Recirculation ON" in label and foggy:
                st.write(f"• ~~{label}~~  → _blocked due to defog risk_")
                st.session_state["__blocked_defog__"] = True
                continue

            col1, col2 = st.columns([4,1])
            with col1:
                count_txt = f" (n={a.get('n', 0)})" if 'n' in a else ""
                st.write(f"• **{label}** → expected impact **{est_txt}**{count_txt}")

            with col2:
                if st.button("Apply", key=f"apply_{a['id']}"):
                    updates = {}
                    if "Increase setpoint" in label:
                        updates["sp_delta"] = min(10, st.session_state.get("sp_delta", 0) + 8)
                    elif "Lower setpoint" in label:
                        updates["sp_delta"] = max(-10, st.session_state.get("sp_delta", 0) - 8)
                    elif "Reduce fan" in label:
                        updates["fan_delta"] = max(-3, st.session_state.get("fan_delta", 0) - 2)
                    elif "Recirculation ON" in label:
                        updates["recirc_on"] = True
                    st.session_state["__pending_apply__"] = updates
                    st.session_state["__notify__"] = f"Applied: {label}"
                    st.session_state["__last_applied_source__"] = "coach"
                    st.rerun()

# ### PHASE 4: Trip Replay with uncertainty band + cumulative Wh
with st.expander("Trip Replay (±150 rows around selection)", expanded=False):
    try:
        w = 150
        i0 = max(0, row_idx - w)
        i1 = min(len(df) - 1, row_idx + w)
        seg = df.iloc[i0:i1+1].copy()

        # Build features once for the window
        Xb = make_features(seg.copy(), setpoint_delta=0, fan_delta=0, recirc_on=False,
                           sensitivity=1.0, predictor=pred, vehicle_type="ICE", proxy_override=proxy_cols)
        yb_series, _ = pred.predict(Xb)

        Xs = make_features(seg.copy(), setpoint_delta=setpoint_delta, fan_delta=fan_delta, recirc_on=recirc_on,
                           sensitivity=sensitivity, predictor=pred, vehicle_type="ICE", proxy_override=proxy_cols)
        ys_series, _ = pred.predict(Xs)

        yb = pd.to_numeric(yb_series, errors="coerce")
        ys = pd.to_numeric(ys_series, errors="coerce")

        # Try to compute per-row quantiles for uncertainty band
        q10 = q90 = None
        try:
            # Prefer a vectorized quantile prediction method if available
            if hasattr(pred, "predict_quantiles"):
                qdict = pred.predict_quantiles(Xb, [0.10, 0.90])  # expected: dict or list-like
                if isinstance(qdict, dict):
                    q10, q90 = pd.to_numeric(pd.Series(qdict.get(0.10, np.nan)), errors="coerce"), pd.to_numeric(pd.Series(qdict.get(0.90, np.nan)), errors="coerce")
                else:
                    q10, q90 = pd.to_numeric(pd.Series(qdict[0]), errors="coerce"), pd.to_numeric(pd.Series(qdict[1]), errors="coerce")
            else:
                # Fallback: if pred.predict returns intervals when asked row-wise (slower)
                lows, highs = [], []
                # only do a light sample to avoid latency if window is huge
                for _i in range(len(Xb)):
                    yi, (lo_i, hi_i) = pred.predict(Xb.iloc[[_i]])
                    lows.append(float(lo_i[0]) if lo_i is not None else np.nan)
                    highs.append(float(hi_i[0]) if hi_i is not None else np.nan)
                q10 = pd.Series(lows, index=yb.index)
                q90 = pd.Series(highs, index=yb.index)
        except Exception:
            q10 = q90 = None

        seg_plot = pd.DataFrame({"Baseline (W)": yb.values, "Simulated (W)": ys.values}, index=seg.index)

        # time axis: use timestamp seconds if present, else assume 1 s cadence
        if "timestamp" in seg.columns:
            t = pd.to_datetime(seg["timestamp"])
            t0 = t.iloc[0]
            t_seconds = (t - t0).dt.total_seconds()
        else:
            t_seconds = np.arange(len(seg_plot), dtype=float)

        # Cumulative Wh saved (assume 1-second sample if no timestamps)
        delta_W = (yb.values - ys.values)
        if "timestamp" in seg.columns:
            # dt between points in hours (fallback to 1-second if any NA)
            dt_hours = np.diff(np.r_[t_seconds.values, t_seconds.values[-1]]) / 3600.0
        else:
            dt_hours = np.ones_like(delta_W) / 3600.0
        delta_Wh_cum = np.cumsum(np.clip(delta_W, a_min=0, a_max=None) * dt_hours)

        # Build charts
        base_df = pd.DataFrame({"t": t_seconds, "Baseline (W)": yb.values})
        sim_df  = pd.DataFrame({"t": t_seconds, "Simulated (W)": ys.values})

        chart_base = alt.Chart(base_df).mark_line().encode(x="t:Q", y=alt.Y("Baseline (W):Q", title="Power (W)"))
        chart_sim  = alt.Chart(sim_df).mark_line(strokeDash=[4,3]).encode(x="t:Q", y="Simulated (W):Q")

        if q10 is not None and q90 is not None and np.isfinite(q10).any() and np.isfinite(q90).any():
            band_df = pd.DataFrame({"t": t_seconds, "low": q10.values, "high": q90.values})
            band = alt.Chart(band_df).mark_area(opacity=0.2).encode(x="t:Q", y="low:Q", y2="high:Q")
            st.altair_chart((band + chart_base + chart_sim).interactive(), use_container_width=True)
            st.caption("Shaded band shows P10–P90 baseline uncertainty (when quantile models are available).")
        else:
            st.altair_chart((chart_base + chart_sim).interactive(), use_container_width=True)
            st.caption("Quantile band unavailable — showing Baseline vs Simulated only.")

        # Cumulative Wh chart
        cum_df = pd.DataFrame({"t": t_seconds, "Cumulative Wh saved": delta_Wh_cum})
        cum_chart = alt.Chart(cum_df).mark_line().encode(x=alt.X("t:Q", title="Time (s)"), y="Cumulative Wh saved:Q")
        st.altair_chart(cum_chart.interactive(), use_container_width=True)

    except Exception as e:
        st.error("Trip Replay failed.")
        st.exception(e)



# --- Context + Expected Save ---
exp_save = float(X_sim.get("_expected_save_W", [0]).iloc[0])
st.write("**Context**")
ctx = (row[["speed","rpm","ambient_temp","cabin_temp"]]
       .rename(columns={"ambient_temp":"Ambient (°C)", "cabin_temp":"Cabin (°C)"})) if "speed" in row.columns and "rpm" in row.columns \
     else row[["ambient_temp","cabin_temp"]].rename(columns={"ambient_temp":"Ambient (°C)", "cabin_temp":"Cabin (°C)"})
st.dataframe(ctx, use_container_width=True)
st.caption(f"Calibrated expected save (info): ~{exp_save:.1f} W (UI guidance only).")

# === Explanation (SHAP) — horizontal, clear labels, padded left ===
st.subheader("Explanation (SHAP)")
if selected_model.startswith("CatBoost"):
    try:
        from catboost import Pool
        cols = list(pred.feature_cols)
        X_cur = X_now.reindex(columns=cols, fill_value=0)
        pool = Pool(X_cur, feature_names=cols)
        shap_vals = pred.model.get_feature_importance(pool, type="ShapValues")
        sv = pd.Series(shap_vals[0][:-1], index=cols)  # exclude bias term
        imp = sv.abs().sort_values(ascending=False).head(10).rename("impact").to_frame().reset_index().rename(columns={"index":"feature"})

        bar = alt.Chart(imp).mark_bar().encode(
            x=alt.X("impact:Q", title="|SHAP| (impact on W)"),
            y=alt.Y("feature:N", sort="-x", title=None)
        )
        st.altair_chart(bar, use_container_width=True)
    except Exception as e:
        st.caption("Live SHAP unavailable for this row/model:")
        st.exception(e)
else:
    st.caption("Explainability: SHAP is shown for CatBoost. For deep models, Integrated Gradients will be added later.")



# ### PHASE 4: Sidebar KPIs (rollups from logs)
with st.sidebar:
    st.markdown("### KPIs (since app start)")
    try:
        log_path = Path("logs/interactions.jsonl")
        if log_path.exists():
            recs = [json.loads(line) for line in log_path.open()]
            log = pd.DataFrame(recs) if recs else pd.DataFrame(columns=[
                "band","saving_W","action_source","blocked_defog","blocked_comfort"
            ])
            if not log.empty:
                by_band = log.groupby("band", dropna=False)["saving_W"].mean().reindex(["heating","mild","cooling"]).rename("Avg saving (W)")
                coach_rate = (log["action_source"].eq("coach").mean()*100.0)
                fog_blocks = int(log.get("blocked_defog", pd.Series([0]*len(log))).sum())
                comfort_blocks = int(log.get("blocked_comfort", pd.Series([0]*len(log))).sum())
                st.metric("Coach apply rate", f"{coach_rate:.0f}%")
                st.dataframe(by_band.fillna(0).to_frame(), use_container_width=True, height=150)
                st.caption(f"Fog blocks: {fog_blocks}  •  Comfort blocks: {comfort_blocks}")
            else:
                st.info("No interactions logged yet.")
        else:
            st.info("`logs/interactions.jsonl` not found yet.")
    except Exception as e:
        st.warning("KPI computation failed.")
        st.exception(e)

# ### PHASE 4: PDF report export (charts + KPIs, matplotlib export for images)
with st.sidebar:
    st.markdown("### Report")
    want_pdf = st.button("Export PDF report")
    if want_pdf:
        export_dir = Path("exports"); export_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = export_dir / f"AC_Advisor_Report_{datetime.datetime.now():%Y%m%d_%H%M}.pdf"
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import cm
            import matplotlib
            matplotlib.use("Agg")          # headless backend
            import matplotlib.pyplot as plt
            import numpy as np

            # --- Re-create trip window data (same as Trip Replay) ---
            w = 150
            i0 = max(0, row_idx - w)
            i1 = min(len(df) - 1, row_idx + w)
            seg = df.iloc[i0:i1+1].copy()

            Xb = make_features(seg.copy(), setpoint_delta=0, fan_delta=0, recirc_on=False,
                               sensitivity=1.0, predictor=pred, vehicle_type="ICE", proxy_override=proxy_cols)
            yb_series, _ = pred.predict(Xb)
            Xs = make_features(seg.copy(), setpoint_delta=setpoint_delta, fan_delta=fan_delta, recirc_on=recirc_on,
                               sensitivity=sensitivity, predictor=pred, vehicle_type="ICE", proxy_override=proxy_cols)
            ys_series, _ = pred.predict(Xs)

            yb = pd.to_numeric(yb_series, errors="coerce").astype(float).values
            ys = pd.to_numeric(ys_series, errors="coerce").astype(float).values
            t_seconds = np.arange(len(yb), dtype=float)
            dt_hours = np.ones_like(yb) / 3600.0
            delta_W = yb - ys
            delta_Wh_cum = np.cumsum(np.clip(delta_W, a_min=0, a_max=None) * dt_hours)

            # --- Save charts as PNGs ---
            p1 = export_dir / "replay_matplotlib.png"
            plt.figure()
            plt.plot(t_seconds, yb, label="Baseline (W)")
            plt.plot(t_seconds, ys, linestyle="--", label="Simulated (W)")
            plt.xlabel("Time (s)"); plt.ylabel("Power (W)"); plt.legend()
            plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()

            p2 = export_dir / "cumulative_wh_matplotlib.png"
            plt.figure()
            plt.plot(t_seconds, delta_Wh_cum)
            plt.xlabel("Time (s)"); plt.ylabel("Cumulative Wh saved")
            plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()

            # --- Build the PDF ---
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            wpg, hpg = A4
            c.setFont("Helvetica-Bold", 14)
            c.drawString(2*cm, hpg-2*cm, "A/C Advisor — Scenario Report")
            c.setFont("Helvetica", 10)
            c.drawString(2*cm, hpg-2.7*cm, f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}")

            kpis = [
                f"Row index: {row_idx}",
                f"Band: {mode_label}",
                f"Ambient / Cabin: {ambient:.1f} °C / {cabin:.1f} °C (ΔT = {dT:+.1f} °C)",
                f"Baseline A/C power: {pred_now:.1f} W",
                f"Simulated A/C power: {pred_sim:.1f} W",
                f"Estimated saving: {saving_W:+.1f} W ({saving_pct:+.1f}%)"
            ]
            ypos = hpg - 4.0*cm
            for k in kpis:
                c.drawString(2*cm, ypos, "• " + k); ypos -= 0.6*cm

            # Embed charts
            c.drawImage(str(p1), 2*cm, ypos-8.0*cm, width=wpg-4*cm, height=8.0*cm, preserveAspectRatio=True)
            ypos -= 8.5*cm
            if ypos < 10*cm:
                c.showPage(); ypos = hpg-3*cm
            c.drawImage(str(p2), 2*cm, ypos-8.0*cm, width=wpg-4*cm, height=8.0*cm, preserveAspectRatio=True)
            c.save()

            st.success(f"Saved: {pdf_path.name}")
            with open(pdf_path, "rb") as f:
                st.download_button("Download report", data=f.read(), file_name=pdf_path.name, use_container_width=True)

        except ImportError:
            st.warning("PDF export requires `reportlab` and `matplotlib`. Install them via requirements.txt.")
        except Exception as e:
            st.error("PDF export failed.")
            st.exception(e)


# --- Log for Coach learning ---
action_src = st.session_state.pop("__last_applied_source__", None) or "manual"
blocked_defog = bool(st.session_state.pop("__blocked_defog__", False))
blocked_comfort = bool(st.session_state.pop("__blocked_comfort__", False))

log_interaction({
    "log_model": selected_model,
    "row_idx": int(row_idx),
    "ambient_c": ambient,
    "cabin_c": cabin,
    "band": band_from_temp(ambient),
    "speed": float(row.get("speed", 0)) if "speed" in row.columns else None,
    "speed_bucket": speed_bucket(float(row.get("speed", 0)) if "speed" in row.columns else None) if "speed" in row.columns else "na",
    "setpoint_delta": int(setpoint_delta),
    "fan_delta": int(fan_delta),
    "recirc_on": bool(recirc_on),
    "sensitivity": float(sensitivity),
    "baseline_W": pred_now,
    "sim_W": pred_sim,
    "saving_W": saving_W,
    "action_source": action_src,
    "blocked_defog": blocked_defog,
    "blocked_comfort": blocked_comfort,
    "proxies": proxy_cols
})



st.markdown("### Model Leaderboard")
lb_rows = []
for name in MODEL_LINEUP:
    m = MODEL_METRICS.get(name, {})
    lb_rows.append({
        "Model": name,
        "MAE (W)": m.get("MAE", "—"),
        "RMSE (W)": m.get("RMSE", "—"),
        "R²": m.get("R2", "—"),
        "Notes": m.get("notes", ""),
        "Status": "Available" if MODEL_AVAILABLE.get(name, False) else "Unavailable",
    })
st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)
st.caption("Notes: ‘Quantile’ powers P10–P90 uncertainty bands. Sequence models use a 32-step window. ‘Experimental’ = preliminary.")

# =========================
# AC Advisor — About (fixed: bg only when expanded)
# =========================
import base64
from pathlib import Path

def _b64safe(p: Path):
    try:
        return base64.b64encode(p.read_bytes()).decode("ascii") if p.exists() else None
    except Exception:
        return None

def render_about_footer():
    assets = Path("assets")
    prof64    = _b64safe(assets / "professor_wang.jpg")
    hemanth64 = _b64safe(assets / "hemanth.jpg")
    kiran64   = _b64safe(assets / "kiranmayee.jpeg")

    # Scoped CSS
    st.markdown("""
    <style>
      .aa-h2 { margin: 0 0 .35rem; font-size: 1.28rem; font-weight: 800; }
      .aa-h3 { margin: .9rem 0 .25rem; font-size: 1.06rem; font-weight: 700; }
      .aa-p  { margin: .2rem 0 .55rem; line-height: 1.55; }
      .aa-ul { margin: .2rem 0 .6rem 1.0rem; }
      .aa-rule { height:1px; background: rgba(120,120,120,.25); border:0; margin: .9rem 0; }
      /* background wrapper only inside expander */
      
      /* team gallery */
      .aa-gallery { display:flex; flex-wrap:wrap; gap:18px; justify-content:center; margin-top:.4rem; }
      .aa-card {
        width: 280px; max-width: 92vw;
        border-radius: 16px; padding: 16px 14px 14px;
        background: rgba(120,120,120,.06);
        border: 1px solid rgba(120,120,120,.18);
        box-shadow: 0 6px 16px rgba(0,0,0,.08);
        text-align:center;
      }
      .aa-avatar {
        width: 118px; height: 118px; border-radius: 50%;
        object-fit: cover; object-position: center;
        border: 3px solid rgba(255,255,255,.55);
        box-shadow: 0 6px 18px rgba(0,0,0,.15);
        margin-bottom: 10px;
      }
      .aa-name { font-weight: 800; font-size: 1.0rem; }
      .aa-role { font-size: .92rem; opacity: .85; margin-top: 2px; }
      .aa-li { text-align:left; margin: 8px auto 0; max-width: 90%; }
      .aa-foot { text-align:center; color:#6b7280; font-size:.92rem; margin-top:.9rem; }
                
                
        .aa-ul {
        list-style: none;
        margin: 0.2rem 0 0.4rem 0;
        padding-left: 0;
        text-align: left;
        }

        .aa-ul li {
        position: relative;
        padding-left: 16px;
        margin: 4px 0;
        line-height: 1.4;
        }

        .aa-ul li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 8px;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #0ea5e9; /* soft cyan accent */
        box-shadow: 0 0 2px rgba(14,165,233,0.4);
        }

    </style>
    """, unsafe_allow_html=True)

    # Expander; set expanded=True if you want it open by default
    with st.expander("About this Project", expanded=False):
        st.markdown("<div class='aa-wrap'>", unsafe_allow_html=True)

        st.markdown("<div class='aa-h2'>AC Advisor</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='aa-p'>AC Advisor is a research-driven application developed at "
            "<b>California State University, Northridge (CSUN)</b>.</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='aa-h3'>Project Goal</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='aa-p'>Develop an intelligent advisor that predicts automobile A/C energy usage from real telemetry "
            "and recommends <b>comfort-aware</b> actions to reduce consumption.</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='aa-h3'>Key Capabilities</div>", unsafe_allow_html=True)
        st.markdown(
            "<ul class='aa-ul'>"
            "<li>CatBoost-based power prediction (baseline vs. simulation)</li>"
            "<li>What-If controls: setpoint Δ, fan Δ, recirculation ON/OFF</li>"
            "<li>AI Coach with comfort & defog guardrails</li>"
            "<li>Trip Replay and cumulative Wh saved visualization</li>"
            "<li>One-click PDF report export with embedded charts</li>"
            "</ul>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='aa-h3'>Supervisor</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='aa-p'><b>Dr. Taehyung (“George”) Wang</b> — Professor, Computer Science, CSUN.</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='aa-h3'>Creators</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='aa-p'><b>Hemanth Kumar Tulabandula</b> &nbsp;|&nbsp; "
            "<b>Kiranmayee Lokam</b> — Graduate Students, CSUN.</div>",
            unsafe_allow_html=True
        )

        

        st.markdown("<hr class='aa-rule'/>", unsafe_allow_html=True)

        # images at the END
        gallery_html = f"""
        <div class="aa-gallery">
          <div class="aa-card">
            {'<img class="aa-avatar" src="data:image/jpeg;base64,'+ (prof64 or '') +'"/>' if prof64 else ''}
            <div class="aa-name">Dr. Taehyung (“George”) Wang</div>
            <div class="aa-role">Professor · Computer Science · CSUN</div>
            <ul class="aa-ul aa-li"><li>Data Mining, Software Engineering, Web Engineering </li></ul>
          </div>
          <div class="aa-card">
            {'<img class="aa-avatar" src="data:image/jpeg;base64,'+ (hemanth64 or '') +'"/>' if hemanth64 else ''}
            <div class="aa-name">Hemanth Kumar Tulabandula</div>
            <div class="aa-role">Graduate Student · CSUN</div>
            <ul class="aa-ul aa-li"><li>Data pipeline, Data preprocessing, Model design, Model validation, UI design, App integration</li></ul>
          </div>
          <div class="aa-card">
            {'<img class="aa-avatar" src="data:image/jpeg;base64,'+ (kiran64 or '') +'"/>' if kiran64 else ''}
            <div class="aa-name">Kiranmayee Lokam</div>
            <div class="aa-role">Graduate Student · CSUN</div>
            <ul class="aa-ul aa-li"><li>Data pipeline, Data preprocessing, Model design, Model validation, UI design, App integration</li></ul>
          </div>
        </div>
        """
        st.markdown(gallery_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close .aa-wrap

    # centered copyright below expander
    st.markdown(
        "<p class='aa-foot'>© 2025 Hemanth Kumar Tulabandula & Kiranmayee Lokam · California State University, Northridge</p>",
        unsafe_allow_html=True
    )


# Render the polished About section
render_about_footer()


