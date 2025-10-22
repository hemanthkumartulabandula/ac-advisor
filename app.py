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

st.set_page_config(page_title="A/C Energy Advisor", page_icon="❄️", layout="wide")

# =========================
# AC Advisor — Header / Banner
# =========================

# Optional local images (add later if you like):
# - Put a square-ish logo at:   assets/ac-advisor-logo.png
# - Or a wide banner at:        assets/banner.png
# Both are optional. Code gracefully falls back to text if not present.

from pathlib import Path

def render_header():
    assets_dir = Path("assets")
    logo_path   = assets_dir / "ac-advisor-logo.png"
    banner_path = assets_dir / "banner.png"

    # Prefer a nice wide banner image if present
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)
    else:
        # Gradient title banner (no image)
        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #0ea5e9, #0369a1);
                padding: 1.2rem 2rem;
                border-radius: 12px;
                color: white;
                text-align: center;
                font-size: 1.8rem;
                font-weight: 650;
                letter-spacing: 0.3px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.18);
                ">
                ❄️ <span style="white-space:nowrap;">AC Advisor</span> — Intelligent Energy Optimization for Automobile A/C
            </div>
            """,
            unsafe_allow_html=True
        )

    # Subline / tagline row with optional small logo on the left
    c1, c2 = st.columns([1, 6], vertical_alignment="center")
    with c1:
        if logo_path.exists():
            st.image(str(logo_path), caption="", width=72)
        else:
            st.markdown("<div style='font-size:2rem;'>❄️</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div style="font-size:1.0rem; color:#475569; line-height:1.45; margin-top:0.2rem;">
                <b>Predict • Simulate • Save</b> — A data-driven advisor that estimates A/C power, 
                explores “what-if” actions (setpoint, fan, recirculation), and quantifies energy savings without compromising comfort.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

# Render it now
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
model_version = st.segmented_control("Model", options=["v1", "v2"], default="v1")
pred = load_predictor(version=model_version)
df = load_data()

st.sidebar.success(f"CatBoost model loaded ✅  ({model_version})")

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
st.info(
    f"Ambient: **{ambient:.1f} °C**  •  ΔT = cabin − ambient = **{dT:+.1f} °C**  •  {hint}\n\n"
    "Band rules: **Heating (≤ 9 °C)** • **Mild (9–15 °C)** • **Cooling (≥ 15 °C)**."
)
with st.expander("What do these bands mean?"):
    st.markdown(
        """
        - **Heating (≤ 9 °C)** → heater-dominant: warmer setpoint costs energy; fan adds a little; recirc benefit small.  
        - **Mild (9–15 °C)** → effects are modest and context-dependent.  
        - **Cooling (≥ 15 °C)** → warmer setpoint and recirc ON reduce A/C load; fan adds a little.
        """
    )

# --- Two-pass predictions ---
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

with st.expander("💡 Coach: best action for this context", expanded=True):
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
with st.expander("📈 Trip Replay (±150 rows around selection)", expanded=False):
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

# --- Debug: show baseline vs simulated proxy values ---
with st.expander("🛠 Debug: baseline vs simulated proxy values", expanded=False):
    sp_col, fan_col, rh_col = proxy_cols["setpoint"], proxy_cols["fan"], proxy_cols["recirc"]
    show_cols = [c for c in [sp_col, fan_col, rh_col] if c in X_sim.columns]
    if show_cols:
        st.write("Columns:", show_cols)
        st.write("Baseline:", X_now[show_cols].iloc[0].to_dict())
        st.write("Simulated:", X_sim[show_cols].iloc[0].to_dict())
    else:
        st.info("Proxy columns not found in X matrices (unexpected).")

# --- Sanity test: prove model reacts to proxy columns ---
with st.expander("🧪 Sanity test: +50% bump on proxy columns", expanded=False):
    import copy
    X_bump = copy.deepcopy(X_now)
    for col in [proxy_cols["setpoint"], proxy_cols["fan"], proxy_cols["recirc"]]:
        if col in X_bump.columns:
            try:
                X_bump[col] = pd.to_numeric(X_bump[col], errors="coerce").fillna(0) * 1.50
            except Exception:
                pass
    y0, _ = pred.predict(X_now); yb, _ = pred.predict(X_bump)
    try:
        delta_bump = float(y0.iloc[0]) - float(yb.iloc[0])
        st.write(f"Δ prediction with +50% bump on proxies: {delta_bump: .4f} W "
                 f"(baseline {float(y0.iloc[0]):.4f} → bumped {float(yb.iloc[0]):.4f})")
        if abs(delta_bump) < 1e-6:
            st.warning("No change detected. Model may not split on these proxies. Consider refreshing SHAP / retraining.")
    except Exception:
        pass

# --- Context + Expected Save ---
exp_save = float(X_sim.get("_expected_save_W", [0]).iloc[0])
st.write("**Context**")
ctx = (row[["speed","rpm","ambient_temp","cabin_temp"]]
       .rename(columns={"ambient_temp":"Ambient (°C)", "cabin_temp":"Cabin (°C)"})) if "speed" in row.columns and "rpm" in row.columns \
     else row[["ambient_temp","cabin_temp"]].rename(columns={"ambient_temp":"Ambient (°C)", "cabin_temp":"Cabin (°C)"})
st.dataframe(ctx, use_container_width=True)
st.caption(f"Calibrated expected save (info): ~{exp_save:.1f} W (UI guidance only).")

# ### PHASE 4: SHAP Top-10 bar (replaces live table)
with st.expander("🔎 Explain this prediction (SHAP)", expanded=False):
    # Offline sample (kept as-is)
    p = Path("models/shap_sample_row.json")
    if p.exists():
        data = json.loads(p.read_text())
        st.write(f"Sample explanation (row {data['row_index']}): Top contributors")
        st.dataframe(pd.DataFrame(data["contribs"]), use_container_width=True)
    else:
        st.info("Run once to generate SHAP outputs: `python -m ac_advisor.shap_analysis`")

    # Live SHAP — Top 10 bar for the selected row
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

# =========================
# AC Advisor — About / Footer
# =========================

def render_about_footer():
    st.markdown("---")
    with st.expander("ℹ️ About this project", expanded=False):
        st.markdown(
            """
            **AC Advisor** is a research-driven application developed at **California State University, Northridge (CSUN)**.

            **Professor / Supervisor**  
            • Dr. Taehyung (“George”) Wang

            **Creators**  
            • Hemanth Kumar Tulabandula  
            • Kiranmayee Lokam

            **Project Goal**  
            Build a practical, explainable assistant that predicts automobile A/C energy usage from real-world telemetry  
            and guides drivers toward safe, comfort-aware energy savings using machine learning and physics-informed rules.

            **Key Capabilities**  
            • CatBoost-based A/C power prediction (baseline vs. simulation)  
            • What-If controls: setpoint Δ, fan Δ, recirculation ON/OFF  
            • AI Coach (context-aware, with guardrails for comfort/defog)  
            • Trip Replay & cumulative Wh saved visualization  
            • One-click PDF report export with embedded charts
            """,
            unsafe_allow_html=True,
        )

    # Tiny centered footer
    st.markdown(
        "<p style='text-align:center; color:#6b7280; font-size:0.90rem; margin-top:1.2rem;'>"
        "© 2025 Hemanth Kumar Tulabandula & Kiranmayee Lokam · California State University, Northridge"
        "</p>",
        unsafe_allow_html=True
    )

# Render it now
render_about_footer()
