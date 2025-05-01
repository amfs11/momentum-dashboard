# view_momentum_dashboard.py

import os
import glob
import pandas as pd
import streamlit as st
from datetime import datetime
import subprocess

# === Setup paths
script_dir = os.path.dirname(__file__)
files = sorted(glob.glob(os.path.join(script_dir, "momentum_predictions_multiday_*.csv")))

st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("📊 Multiday Momentum Dashboard")

# === Load data
if not files:
    st.warning("No prediction files found.")
    st.stop()

latest_file = files[-1]
df = pd.read_csv(latest_file)

if df.empty:
    st.warning("Prediction file is empty.")
    st.stop()

# === Interpretation Logic
def interpret_signal(row):
    t1, t3, t5 = row["pred_t+1"], row["pred_t+3"], row["pred_t+5"]
    if t1 > 0 and t3 > 0 and t5 > 0:
        return "Strong Buy & Hold (5 days)"
    elif t1 > 0 and t3 > 0 and t5 < 0:
        return "Buy Short-Term (1-3 days)"
    elif t1 > 0 and t3 < 0 and t5 < 0:
        return "Quick Trade (1-day pop)"
    elif t1 < 0 and t3 < 0 and t5 < 0:
        return "Avoid / No Signal"
    elif t1 < 0 and t3 > 0 and t5 > 0:
        return "Wait & Watch"
    else:
        return "Mixed Signal - Use Caution"

df["signal"] = df.apply(interpret_signal, axis=1)

# === Sidebar Filters
st.sidebar.header("📋 Filter Options")
min_conf = st.sidebar.slider("Minimum Confidence (%)", 0.0, 100.0, 2.0, 0.5)

signal_types = st.sidebar.multiselect(
    "Signal Type", df["signal"].unique(), default=df["signal"].unique()
)

tickers_available = sorted(df["ticker"].unique())
selected_tickers = st.sidebar.multiselect(
    "Filter by Ticker", tickers_available, default=tickers_available
)

filtered = df[
    (df["confidence"] >= min_conf) &
    (df["signal"].isin(signal_types)) &
    (df["ticker"].isin(selected_tickers))
]

# === Add Color Coding
signal_color_map = {
    "Strong Buy & Hold (5 days)": "🟢",
    "Buy Short-Term (1-3 days)": "🟢",
    "Quick Trade (1-day pop)": "🟡",
    "Wait & Watch": "🟠",
    "Mixed Signal - Use Caution": "⚠️",
    "Avoid / No Signal": "🔴"
}

styled_df = filtered.copy()
styled_df["signal_icon"] = styled_df["signal"].map(signal_color_map)
styled_df["signal"] = styled_df["signal_icon"] + " " + styled_df["signal"]
styled_df.drop(columns=["signal_icon"], inplace=True)

# === Display
st.subheader(f"📈 Predictions for {df['date'].iloc[0]}")
st.dataframe(styled_df.sort_values("confidence", ascending=False).reset_index(drop=True))

# === Show Chart for Selected Ticker
import altair as alt

st.markdown("---")
st.subheader("📈 Price History")

selected_ticker = st.selectbox("Choose a ticker to view chart", sorted(df["ticker"].unique()))

# Load historical price data
price_path = os.path.join(script_dir, "historical_prices.csv")
if os.path.exists(price_path):
    prices_df = pd.read_csv(price_path)
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    chart_df = prices_df[(prices_df["ticker"] == selected_ticker)].sort_values("date").copy()

    if not chart_df.empty:
        st.line_chart(chart_df.set_index("date")["close"])
    else:
        st.warning(f"No price data found for {selected_ticker}")
else:
    st.error("historical_prices.csv not found.")

# === Download
st.download_button(
    label="📥 Download CSV",
    data=styled_df.to_csv(index=False).encode("utf-8"),
    file_name=f"filtered_predictions_{datetime.now().date()}.csv",
    mime="text/csv"
)

# === Manual Email Trigger
st.markdown("---")
if st.button("📤 Send Momentum Report Now"):
    try:
        subprocess.run(["python", "scripts/send_momentum_report.py"], check=True)
        st.success("✅ Report sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send report: {e}")
