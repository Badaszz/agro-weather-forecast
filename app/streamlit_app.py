import sys
import os

# Resolve project root regardless of where the app is launched from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # ensures all relative paths like "data/..." resolve correctly

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import date, timedelta

from src.database import (
    get_predictions,
    get_actuals,
    get_predictions_with_actuals,
    get_monitoring_results,
)
from src.monitor import get_latest_monitoring_result, BASELINE_MAE
from src.predict import predict_next_day_rainfall

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Lagos Rainfall Forecast",
    page_icon="🌧️",
    layout="wide",
)

QUALITY_LOG_PATH = "data/quality_results.json"
MONITOR_LOG_PATH = "data/monitor_log.json"


# === HELPERS ===
def load_json_log(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def season_label(month: int) -> str:
    if month in [6, 9, 10]:  return "🌧️ Peak Rain"
    if month in [4, 5, 7]:   return "🌦️ Wet Season"
    if month == 8:            return "☁️ August Break"
    return "☀️ Dry Season"


# === HEADER ===
st.title("🌾 Lagos Agricultural Weather Forecast")
st.caption("Automated daily rainfall predictions for agricultural planning · Lagos, Nigeria")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🌧️  Today's Forecast",
    "📈  Model Accuracy",
    "🔍  Data Quality & Monitoring",
])


# ================================================================
# TAB 1 — TODAY'S FORECAST
# ================================================================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Today's Prediction")

        with st.spinner("Fetching latest forecast..."):
            try:
                result = predict_next_day_rainfall(6.5244, 3.3792)
                predicted = result["predicted_rainfall_mm"]
                pred_date = result["prediction_date"]
                based_on  = result["based_on_data_up_to"]

                # Rainfall category
                if predicted == 0:
                    rain_label, rain_color, rain_icon = "No Rain", "green",  "☀️"
                elif predicted < 2:
                    rain_label, rain_color, rain_icon = "Light Rain",   "blue",   "🌦️"
                elif predicted < 10:
                    rain_label, rain_color, rain_icon = "Moderate Rain","orange", "🌧️"
                else:
                    rain_label, rain_color, rain_icon = "Heavy Rain",   "red",    "⛈️"

                st.metric(
                    label=f"{rain_icon} Predicted Rainfall",
                    value=f"{predicted} mm",
                    help="Predicted using Ridge Regression (α=100)"
                )
                st.markdown(f"**Condition:** :{rain_color}[{rain_label}]")
                st.markdown(f"**Prediction date:** {pred_date}")
                st.markdown(f"**Based on data up to:** {based_on}")

                # Season
                month = date.today().month
                st.markdown(f"**Current season:** {season_label(month)}")
                st.markdown(f"**Model:** {result['model']}")

            except Exception as e:
                st.error(f"Could not fetch prediction: {e}")

    with col2:
        st.subheader("14-Day Prediction History")
        preds = get_predictions(days=14)

        if preds:
            df_preds = pd.DataFrame(preds)
            df_preds["prediction_date"] = pd.to_datetime(df_preds["prediction_date"])

            actuals = get_actuals(days=14)
            df_acts = pd.DataFrame(actuals) if actuals else pd.DataFrame(
                columns=["actual_date", "actual_rainfall"]
            )
            if not df_acts.empty:
                df_acts["actual_date"] = pd.to_datetime(df_acts["actual_date"])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_preds["prediction_date"],
                y=df_preds["predicted_rainfall_mm"],
                name="Predicted",
                marker_color="royalblue",
                opacity=0.7,
            ))
            if not df_acts.empty:
                fig.add_trace(go.Scatter(
                    x=df_acts["actual_date"],
                    y=df_acts["actual_rainfall"],
                    name="Actual",
                    mode="lines+markers",
                    line=dict(color="tomato", width=2),
                    marker=dict(size=7),
                ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Rainfall (mm)",
                legend=dict(orientation="h"),
                height=350,
                margin=dict(t=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction history yet. Check back after the first daily run.")

    # Recent predictions table
    st.subheader("Recent Predictions Log")
    all_preds = get_predictions(days=30)
    all_acts  = get_actuals(days=30)

    if all_preds:
        df_p = pd.DataFrame(all_preds)
        df_a = pd.DataFrame(all_acts) if all_acts else pd.DataFrame(
            columns=["actual_date", "actual_rainfall"]
        )
        if not df_a.empty:
            df_merged = df_p.merge(
                df_a.rename(columns={"actual_date": "prediction_date"}),
                on="prediction_date", how="left"
            )
            df_merged["error_mm"] = (
                df_merged["actual_rainfall"] - df_merged["predicted_rainfall_mm"]
            ).round(2)
        else:
            df_merged = df_p.copy()
            df_merged["actual_rainfall"] = "Pending"
            df_merged["error_mm"]        = "—"

        st.dataframe(
            df_merged[[
                "prediction_date", "predicted_rainfall_mm",
                "actual_rainfall", "error_mm", "model_version"
            ]].sort_values("prediction_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No predictions logged yet.")


# ================================================================
# TAB 2 — MODEL ACCURACY
# ================================================================
with tab2:
    pairs = get_predictions_with_actuals(days=90)

    if len(pairs) < 3:
        st.info("Not enough paired predictions + actuals yet. "
                "Come back after at least 7 days of daily runs.")
    else:
        df = pd.DataFrame(pairs)
        df["prediction_date"] = pd.to_datetime(df["prediction_date"])
        df["error"]           = (df["actual_rainfall"] - df["predicted_rainfall_mm"]).abs()
        df["month"]           = df["prediction_date"].dt.month
        df["rolling_mae_7d"]  = df["error"].rolling(7, min_periods=3).mean()

        overall_mae  = round(df["error"].mean(), 4)
        overall_rmse = round(
            (((df["actual_rainfall"] - df["predicted_rainfall_mm"]) ** 2).mean()) ** 0.5, 4
        )

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall MAE",      f"{overall_mae} mm")
        k2.metric("Overall RMSE",     f"{overall_rmse} mm")
        k3.metric("Baseline MAE",     f"{BASELINE_MAE} mm")
        delta = round(overall_mae - BASELINE_MAE, 4)
        k4.metric("vs Baseline",      f"{delta:+.4f} mm",
                  delta_color="inverse")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted vs Actual")
            fig = px.scatter(
                df,
                x="actual_rainfall",
                y="predicted_rainfall_mm",
                trendline="ols",
                labels={
                    "actual_rainfall":        "Actual (mm)",
                    "predicted_rainfall_mm":  "Predicted (mm)",
                },
                color_discrete_sequence=["royalblue"],
            )
            max_val = max(df["actual_rainfall"].max(),
                          df["predicted_rainfall_mm"].max()) + 1
            fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(color="red", dash="dash"))
            fig.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Rolling 7-Day MAE Over Time")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["prediction_date"],
                y=df["rolling_mae_7d"],
                mode="lines",
                name="Rolling MAE (7d)",
                line=dict(color="royalblue", width=2),
                fill="tozeroy",
                fillcolor="rgba(65,105,225,0.1)",
            ))
            fig2.add_hline(
                y=BASELINE_MAE,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Baseline MAE: {BASELINE_MAE}",
            )
            drift_threshold = round(BASELINE_MAE * 1.3, 4)
            fig2.add_hline(
                y=drift_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Drift threshold: {drift_threshold}",
            )
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="MAE (mm)",
                height=350,
                margin=dict(t=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Season breakdown
        st.subheader("Performance by Season")
        season_map = {
            "peak_rain":  [6, 9, 10],
            "wet_season": [4, 5, 7],
            "aug_break":  [8],
            "dry_season": [11, 12, 1, 2, 3],
        }
        season_rows = []
        for season, months in season_map.items():
            subset = df[df["month"].isin(months)]
            if len(subset) >= 2:
                season_rows.append({
                    "Season":       season.replace("_", " ").title(),
                    "Samples":      len(subset),
                    "MAE (mm)":     round(subset["error"].mean(), 4),
                    "RMSE (mm)":    round(
                        (((subset["actual_rainfall"] -
                           subset["predicted_rainfall_mm"]) ** 2).mean()) ** 0.5, 4
                    ),
                    "Avg Actual (mm)": round(subset["actual_rainfall"].mean(), 2),
                })
        if season_rows:
            st.dataframe(
                pd.DataFrame(season_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Not enough seasonal data yet.")

        # Error distribution
        st.subheader("Prediction Error Distribution")
        df["signed_error"] = df["actual_rainfall"] - df["predicted_rainfall_mm"]
        fig3 = px.histogram(
            df,
            x="signed_error",
            nbins=30,
            labels={"signed_error": "Error (Actual − Predicted) mm"},
            color_discrete_sequence=["royalblue"],
        )
        fig3.add_vline(x=0, line_dash="dash", line_color="red",
                       annotation_text="Zero error")
        fig3.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)


# ================================================================
# TAB 3 — DATA QUALITY & MONITORING
# ================================================================
with tab3:
    col1, col2 = st.columns(2)

    # --- Quality checks ---
    with col1:
        st.subheader("🔍 Latest Data Quality Report")
        quality_history = load_json_log(QUALITY_LOG_PATH)

        if not quality_history:
            st.info("No quality checks run yet. Run `uv run python -m src.quality` first.")
        else:
            latest_q = quality_history[-1]
            overall  = latest_q["overall"]

            if overall == "PASS":
                st.success(f"✅ PASS — {latest_q['passed']}/{latest_q['total']} checks passed")
            else:
                st.error(f"❌ FAIL — {latest_q['failed']} check(s) failed")

            st.caption(f"Last run: {latest_q['run_at']}")

            rows = []
            for check_name, result in latest_q["checks"].items():
                rows.append({
                    "Check":   check_name.replace("_", " ").title(),
                    "Status":  "✅ Pass" if result["passed"] else "❌ Fail",
                    "Detail":  result["detail"],
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            # Quality history chart
            if len(quality_history) > 1:
                st.subheader("Quality Check History")
                hist_rows = [
                    {
                        "run_at":  h["run_at"],
                        "passed":  h["passed"],
                        "failed":  h["failed"],
                        "overall": h["overall"],
                    }
                    for h in quality_history
                ]
                df_qh = pd.DataFrame(hist_rows)
                fig_q  = go.Figure()
                fig_q.add_trace(go.Bar(
                    x=df_qh["run_at"], y=df_qh["passed"],
                    name="Passed", marker_color="green"
                ))
                fig_q.add_trace(go.Bar(
                    x=df_qh["run_at"], y=df_qh["failed"],
                    name="Failed", marker_color="red"
                ))
                fig_q.update_layout(
                    barmode="stack",
                    xaxis_title="Run date",
                    yaxis_title="Checks",
                    height=280,
                    margin=dict(t=20),
                )
                st.plotly_chart(fig_q, use_container_width=True)

    # --- Monitoring / drift ---
    with col2:
        st.subheader("📊 Model Drift Monitor")
        latest_m = get_latest_monitoring_result()

        if not latest_m:
            st.info("No monitoring results yet. Need at least 7 days of "
                    "paired predictions + actuals.")
        else:
            drift = latest_m["drift"]
            if drift["drift_detected"]:
                st.error("🚨 Drift detected — model performance has degraded!")
            else:
                st.success("✅ No drift detected — model performing normally")

            m1, m2, m3 = st.columns(3)
            m1.metric("Current MAE",    f"{drift['current_mae']} mm")
            m2.metric("Baseline MAE",   f"{drift['baseline_mae']} mm")
            m3.metric("Degradation",    f"{drift['degradation_pct']:+.1f}%",
                      delta_color="inverse")

            st.caption(f"Evaluated on: {latest_m['evaluated_on']} "
                       f"· {latest_m['num_samples']} samples")

            # Rolling MAE chart from monitor log
            if latest_m.get("rolling_mae_7d"):
                df_roll = pd.DataFrame(latest_m["rolling_mae_7d"])
                df_roll["prediction_date"] = pd.to_datetime(df_roll["prediction_date"])
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=df_roll["prediction_date"],
                    y=df_roll["rolling_mae_7d"],
                    mode="lines+markers",
                    name="Rolling MAE",
                    line=dict(color="royalblue"),
                ))
                fig_d.add_hline(
                    y=BASELINE_MAE * 1.3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Drift threshold",
                )
                fig_d.update_layout(
                    xaxis_title="Date",
                    yaxis_title="MAE (mm)",
                    height=280,
                    margin=dict(t=20),
                )
                st.plotly_chart(fig_d, use_container_width=True)

            # Season breakdown from latest monitor result
            season_data = latest_m.get("season_metrics", {})
            season_rows = []
            for season, sm in season_data.items():
                if sm:
                    season_rows.append({
                        "Season":   season.replace("_", " ").title(),
                        "MAE (mm)": sm["mae"],
                        "RMSE (mm)":sm["rmse"],
                    })
            if season_rows:
                st.subheader("Season Performance")
                st.dataframe(
                    pd.DataFrame(season_rows),
                    use_container_width=True,
                    hide_index=True,
                )

    # --- DB stats ---
    st.divider()
    st.subheader("🗄️ Database Stats")
    d1, d2, d3, d4 = st.columns(4)
    all_p  = get_predictions(days=365)
    all_a  = get_actuals(days=365)
    all_mo = get_monitoring_results(days=365)
    paired = get_predictions_with_actuals(days=365)

    d1.metric("Total Predictions",    len(all_p))
    d2.metric("Total Actuals",        len(all_a))
    d3.metric("Paired (evaluable)",   len(paired))
    d4.metric("Monitoring runs",      len(all_mo))