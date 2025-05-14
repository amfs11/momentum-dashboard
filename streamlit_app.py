import streamlit as st
import pandas as pd
import os
from datetime import date
import matplotlib.pyplot as plt

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="Quant Dashboard", layout="wide")

# === Secure Access ===
password = st.text_input("🔒 Enter dashboard password", type="password")
stored_pw = st.secrets.get("DASHBOARD_PASSWORD", "")

if password != stored_pw:
    st.warning("Incorrect password or not provided.")
    st.stop()

st.title("📊 Quant Forecast & Backtest Dashboard")

# === Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "Forecast", "Accuracy", "Backtest", "Weekly Hit Rate", "Model Comparison",
    "Model Sharpe Trend", "Dashboard", "Health", "Logs", "Metrics", 
    "Realized P&L", "Unrealized P&L", "Backtest Summary"
])

# === Tab 1: Forecast ===
with tab1:
    st.header("📈 Latest Forecast")

    # ✅ Manual Refresh
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    latest_date = date.today().strftime("%Y-%m-%d")
    forecast_path = f"scripts/core/forecast_{latest_date}.csv"

    if os.path.exists(forecast_path):
        forecast_df = pd.read_csv(forecast_path)

        # === Ticker Filter ===
        tickers = sorted(forecast_df["ticker"].unique())
        selected = st.multiselect(
            "Select one or more tickers to view:",
            options=tickers,
            default=tickers
        )
        filtered_df = forecast_df[forecast_df["ticker"].isin(selected)]

        # === Download Button ===
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast (Filtered)",
            data=csv,
            file_name=f"forecast_{latest_date}.csv",
            mime="text/csv"
        )

        # === Show Filtered Forecast ===
        st.dataframe(filtered_df.sort_values("pct_pred", ascending=False))
    else:
        st.warning("No forecast available for today.")

# === Tab 2: Backtest Summary ===
with tab2:
    st.header("🧪 Backtest Summary")
    bt_path = "scripts/logs/backtest_results.csv"
    if os.path.exists(bt_path):
        bt_df = pd.read_csv(bt_path)
        st.metric("Sample Size", len(bt_df))
        st.metric("Hit Rate", f"{(bt_df['hit'].mean() * 100):.2f}%")
        st.metric("Avg Return", f"{bt_df['actual_return'].mean():.2f}%")
    else:
        st.warning("Backtest results not found.")

# === Tab 3: Per-Ticker Metrics ===
with tab3:
    st.header("🏅 Per-Ticker Backtest Performance")

    ticker_path = "scripts/logs/backtest_per_ticker.csv"
    if os.path.exists(ticker_path):
        ticker_df = pd.read_csv(ticker_path)

        # === Ticker Filter ===
        all_tickers = sorted(ticker_df["ticker"].unique())
        selected_tickers = st.multiselect(
            "Select one or more tickers to analyze:",
            options=all_tickers,
            default=all_tickers  # show all by default
        )

        filtered_df = ticker_df[ticker_df["ticker"].isin(selected_tickers)]

        # === Table ===
        st.dataframe(filtered_df.sort_values("sharpe", ascending=False))

        # === Download Button ===
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Ticker Metrics",
            data=csv,
            file_name="filtered_per_ticker.csv",
            mime="text/csv"
        )

        # === Toggle Metric Chart ===
        st.subheader("📊 Ticker Accuracy & Risk Visualization")
        metric_option = st.selectbox("Select metric to visualize:", ["Hit Rate (%)", "Sharpe Ratio"])

        if metric_option == "Hit Rate (%)":
            chart_data = filtered_df.sort_values("hit_rate", ascending=True)
            st.bar_chart(chart_data.set_index("ticker")["hit_rate"] * 100)
        else:
            chart_data = filtered_df.sort_values("sharpe", ascending=True)
            st.bar_chart(chart_data.set_index("ticker")["sharpe"])
    else:
        st.warning("Per-ticker backtest not found.")

# === Tab 4: Monthly Sharpe Plot ===
with tab4:
    st.header("📈 Monthly Sharpe Plot")
    plot_path = "scripts/logs/monthly_sharpe_plot.png"
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        st.warning("Sharpe plot not found.")

# === Tab 5: Weekly Prediction Hit Rate ===
with tab5:
    st.header("📊 Weekly Prediction Hit Rate")
    eval_log_path = "scripts/logs/live_accuracy_log.csv"
    if os.path.exists(eval_log_path):
        df = pd.read_csv(eval_log_path)
        df["date"] = pd.to_datetime(df["date"])
        df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

        weekly_hit = df.groupby("week")["hit"].mean().reset_index()
        weekly_hit["hit_rate"] = (weekly_hit["hit"] * 100).round(2)

        st.line_chart(weekly_hit.set_index("week")["hit_rate"])
        st.dataframe(
            weekly_hit.rename(columns={"week": "Week", "hit_rate": "Hit Rate (%)"})
        )
    else:
        st.warning("⚠️ Weekly evaluation log not found (live_accuracy_log.csv).")

# === Tab 6: Ticker Profile
with tab6:
    st.header("🔎 Ticker Profile Explorer")

    bt_path = "scripts/logs/backtest_per_ticker.csv"
    forecast_path = f"scripts/core/forecast_{latest_date}.csv"
    eval_log = "scripts/logs/live_accuracy_log.csv"

    # === Ticker Selection ===
    all_tickers = []
    if os.path.exists(bt_path):
        all_tickers = list(pd.read_csv(bt_path)["ticker"].unique())
    elif os.path.exists(forecast_path):
        all_tickers = list(pd.read_csv(forecast_path)["ticker"].unique())

    if all_tickers:
        selected_ticker = st.selectbox("Select a ticker to profile:", sorted(all_tickers))

        # === Forecast Details ===
        st.subheader("🧠 Latest Forecast")
        if os.path.exists(forecast_path):
            df = pd.read_csv(forecast_path)
            df = df[df["ticker"] == selected_ticker]
            st.dataframe(df.astype(str))
        else:
            st.info("No forecast data available.")

        # === Backtest Summary ===
        st.subheader("📈 Backtest Metrics")
        if os.path.exists(bt_path):
            bt_df = pd.read_csv(bt_path)
            row = bt_df[bt_df["ticker"] == selected_ticker]
            if not row.empty:
                st.write(row.T.astype(str))
            else:
                st.info("No backtest available for this ticker.")

        # === Live Accuracy Log ===
        st.subheader("📊 Recent Evaluation Hits")
        if os.path.exists(eval_log):
            eval_df = pd.read_csv(eval_log)
            eval_df = eval_df[eval_df["ticker"] == selected_ticker]
            eval_df["date"] = pd.to_datetime(eval_df["date"])
            if not eval_df.empty:
                st.line_chart(eval_df.set_index("date")["hit"])
            else:
                st.info("No evaluation data available for this ticker.")
        else:
            st.info("Evaluation log not available.")

        # === Confidence Over Time (pct_pred) ===
        st.subheader("📈 Predicted Confidence Over Time (pct_pred)")
        if os.path.exists(forecast_path):
            df_all = pd.read_csv(forecast_path)
            if "pct_pred" in df_all.columns:
                df_all["date"] = pd.to_datetime(df_all.get("date", latest_date))
                df_ticker = df_all[df_all["ticker"] == selected_ticker]
                if not df_ticker.empty:
                    st.line_chart(df_ticker.set_index("date")["pct_pred"])
                else:
                    st.info("No forecast entries for this ticker.")
            else:
                st.info("No 'pct_pred' column found in forecast.")
        else:
            st.info("No forecast file found to show prediction confidence.")
    else:
        st.info("No tickers found in forecast or backtest.")

# === Tab 7: Forecast Trends Over Time
with tab7:
    st.header("📈 Forecast Trends Over Time")

    history_path = "scripts/logs/forecast_history.csv"
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])

        tickers = sorted(df["ticker"].unique())
        selected = st.selectbox("Select a ticker to plot:", tickers)

        df_ticker = df[df["ticker"] == selected].sort_values("forecast_date")

        # Plot confidence (pct_pred) over time
        st.subheader("📊 Model Confidence (pct_pred)")
        st.line_chart(df_ticker.set_index("forecast_date")["pct_pred"])

        # Optional: Plot label counts over time
        st.subheader("🔍 Signal History (Buy / Avoid)")
        label_counts = df_ticker.groupby(["forecast_date", "label"]).size().unstack(fill_value=0)
        st.bar_chart(label_counts)
    else:
        st.warning("❌ forecast_history.csv not found. Run build_forecast_history.py first.")

# === Tab 8: Forecast Accuracy
with tab8:
    st.header("📊 Forecast Accuracy Analysis")

    accuracy_path = "scripts/logs/forecast_accuracy_log.csv"
    if os.path.exists(accuracy_path):
        import matplotlib.pyplot as plt

        df = pd.read_csv(accuracy_path)
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])

        # === Directional Accuracy Over Time ===
        st.subheader("📈 Rolling Hit Rate")
        weekly = df.groupby(df["forecast_date"].dt.to_period("W"))["hit"].mean().reset_index()
        weekly["forecast_date"] = weekly["forecast_date"].astype(str)
        weekly["hit_rate"] = (weekly["hit"] * 100).round(2)
        st.line_chart(weekly.set_index("forecast_date")["hit_rate"])

        # === MAE (Mean Absolute Error) Over Time ===
        st.subheader("📉 Mean Absolute Error (MAE)")
        df["abs_error"] = df["error"].abs()
        mae_weekly = df.groupby(df["forecast_date"].dt.to_period("W"))["abs_error"].mean().reset_index()
        mae_weekly["forecast_date"] = mae_weekly["forecast_date"].astype(str)
        st.line_chart(mae_weekly.set_index("forecast_date")["abs_error"])

        # === Error Distribution ===
        st.subheader("🧠 Error Distribution (All Forecasts)")
        fig, ax = plt.subplots()
        ax.hist(df["error"].dropna(), bins=30, color="#4CAF50", edgecolor="black")
        ax.set_title("Forecast Error Distribution")
        ax.set_xlabel("Prediction Error (actual - predicted)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # === Per-Ticker Hit Rate ===
        st.subheader("🏅 Per-Ticker Accuracy")
        per_ticker = df.groupby("ticker")["hit"].mean().reset_index()
        per_ticker["hit_rate"] = (per_ticker["hit"] * 100).round(2)
        st.bar_chart(per_ticker.set_index("ticker")["hit_rate"])

        # === Actual vs Predicted Scatterplot ===
        st.subheader("📈 Actual vs Predicted Scatterplot")
        scatter_df = df.dropna(subset=["pct_pred", "actual_return_3d"])
        if not scatter_df.empty:
            fig2, ax2 = plt.subplots()
            ax2.scatter(scatter_df["pct_pred"], scatter_df["actual_return_3d"],
                        alpha=0.5, color="#2196F3", edgecolor="black")
            ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax2.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax2.set_title("Actual vs Predicted 3-Day Returns")
            ax2.set_xlabel("Predicted Return (pct_pred)")
            ax2.set_ylabel("Actual Return (3-day)")
            st.pyplot(fig2)
        else:
            st.info("Not enough data for scatterplot.")
    else:
        st.warning("❌ forecast_accuracy_log.csv not found. Run evaluate_forecast_accuracy.py first.")
 
# === Tab 9: Model Comparison
with tab9:
    st.header("🧠 Model Comparison Summary")

    models = {
        "XGBoost": "scripts/logs/walk_xgb.csv",
        "LightGBM": "scripts/logs/walk_lgbm.csv",
        "CatBoost": "scripts/logs/walk_catboost.csv",
        "Ensemble": "scripts/logs/walk_ensemble.csv"
    }

    summary_rows = []

    for name, path in models.items():
        if os.path.exists(path):
            df_model = pd.read_csv(path)
            avg_sharpe = df_model["sharpe"].mean()
            avg_mae = df_model["mae"].mean()
            avg_rmse = df_model["rmse"].mean()
            summary_rows.append({
                "Model": name,
                "Avg Sharpe": round(avg_sharpe, 2),
                "Avg MAE": round(avg_mae, 4),
                "Avg RMSE": round(avg_rmse, 4)
            })
        else:
            st.warning(f"❗ {name} log not found at {path}")

    if summary_rows:
       summary_df = pd.DataFrame(summary_rows)

    # ✅ Highlight the top Sharpe model
    # top_model = summary_df.loc[summary_df["Avg Sharpe"].idxmax(), "Model"]
    # Optional highlighting of top model – disabled for now
    # summary_df["🏆 Top Performer"] = summary_df["Model"].apply(lambda m: "✅" if m == top_model else "")

    # st.dataframe(summary_df.set_index("Model"))

# === Tab 10: Model Sharpe Trend
with tab10:
    st.header("📈 Model Sharpe Ratio Over Time")

    models = {
        "XGBoost": "scripts/logs/walk_xgb.csv",
        "LightGBM": "scripts/logs/walk_lgbm.csv",
        "CatBoost": "scripts/logs/walk_catboost.csv",
        "Ensemble": "scripts/logs/walk_ensemble.csv"
    }

    for name, path in models.items():
        if os.path.exists(path):
            df_model = pd.read_csv(path)
            if "start_date" in df_model.columns:
                df_model["start_date"] = pd.to_datetime(df_model["start_date"])
                weekly = df_model.groupby(df_model["start_date"].dt.to_period("W"))["sharpe"].mean().reset_index()
                weekly["start_date"] = weekly["start_date"].astype(str)

                st.subheader(f"{name} – Weekly Sharpe Trend")
                st.line_chart(weekly.set_index("start_date")["sharpe"])

                # ✅ Add Sharpe trend note
                recent = weekly["sharpe"].tail(3).values
                if len(recent) == 3:
                    if recent[-1] > recent[0]:
                        st.caption("⬆️ Sharpe trend improving")
                    else:
                        st.caption("⬇️ Sharpe trend weakening")
            else:
                st.warning(f"{name}: 'start_date' column missing")
        else:
            st.warning(f"{name} log not found at {path}")

# === Tab 11: Realized P&L
with tab11:
    st.subheader("📈 Realized P&L – Closed Trades")
    pnl_path = "scripts/logs/pnl_log.csv"

    if not os.path.exists(pnl_path):
        st.info("No P&L data available yet.")
    else:
        df = pd.read_csv(pnl_path)
        df["buy_time"] = pd.to_datetime(df["buy_time"])
        df["sell_time"] = pd.to_datetime(df["sell_time"])
        df = df.sort_values("sell_time", ascending=False)

        st.metric("Total Trades", len(df))
        st.metric("Total Net PnL ($)", round(df["pnl_dollars"].sum(), 2))
        win_rate = (df["pnl_dollars"] > 0).mean()
        st.metric("Win Rate (%)", f"{win_rate:.2%}")

        st.dataframe(df[[
            "ticker", "qty", "entry_price", "exit_price",
            "pnl_dollars", "pnl_percent", "holding_days",
            "buy_time", "sell_time"
        ]], use_container_width=True)
 
 # === Tab 12: Unrealized P&L
with tab12:
    st.subheader("📉 Unrealized P&L – Open Positions")
    pnl_path = "scripts/logs/unrealized_pnl.csv"

    if not os.path.exists(pnl_path):
        st.info("No unrealized P&L data available yet.")
    else:
        df = pd.read_csv(pnl_path)
        df = df.sort_values("unrealized_dollars", ascending=False)

        st.metric("Open Positions", len(df))
        st.metric("Total Unrealized PnL ($)", round(df["unrealized_dollars"].sum(), 2))
        avg_holding = df["holding_days"].mean()
        st.metric("Avg Holding Days", f"{avg_holding:.2f}")

        st.dataframe(df[[
            "ticker", "qty", "entry_price", "current_price",
            "unrealized_dollars", "unrealized_percent", "holding_days"
        ]], use_container_width=True)

# === Tab 13: Backtest Summary
with tab13:
    st.subheader("📊 Backtest Summary")

    sharpe_path = "scripts/logs/monthly_sharpe_plot.png"
    monthly_summary_path = "scripts/logs/backtest_monthly_summary.csv"

    if os.path.exists(sharpe_path):
        st.image(sharpe_path, caption="Monthly Sharpe Ratios")

    if os.path.exists(monthly_summary_path):
        summary_df = pd.read_csv(monthly_summary_path)
        st.dataframe(summary_df, use_container_width=True)
        st.metric("Avg Sharpe", round(summary_df["sharpe"].mean(), 2))
        st.metric("Hit Rate", f"{summary_df['hit_rate'].mean():.2%}")
