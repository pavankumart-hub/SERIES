# 📊 Unified ARIMA + Polynomial Detrending Dashboard
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from numpy.polynomial import Polynomial
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA + Polynomial Detrending Dashboard", layout="wide")
st.title("📈 ARIMA + Polynomial Detrending Dashboard (Original Scale Forecast)")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
run_btn = st.sidebar.button("Run Analysis")

# --- Function Definitions ---
def kpss_test(series):
    statistic, p_value, _, _ = kpss(series, regression="c", nlags="auto")
    return p_value

# --- Main Analysis ---
if run_btn:
    progress_bar = st.progress(0)
    progress_text = st.empty()

    try:
        progress_text.text("Downloading data...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the given ticker or range.")
            st.stop()

        progress_bar.progress(10)
        progress_text.text("Data downloaded successfully...")

        # --- Basic Info ---
        st.subheader("📋 Company Information")
        st.write(f"**Ticker:** {ticker}")
        st.write(f"**Number of data points collected:** {len(data)}")
        st.write(f"**Date Range:** {start_date} to {end_date}")

        symbol = "₹" if ticker.endswith(".NS") else "$"

        # --- Plot All Price Series ---
        st.subheader("📊 Price Series Overview")
        fig, ax = plt.subplots(figsize=(10, 4))
        data[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
        ax.set_title("All Price Series (Open, High, Low, Close)")
        ax.set_ylabel(f"Price ({symbol})")
        st.pyplot(fig)

        progress_bar.progress(20)
        progress_text.text("Starting ARIMA analysis for Open, High, Low, Close...")

        # --- Run for all series ---
        summary_rows = []

        for col in ['Open', 'High', 'Low', 'Close']:
            st.subheader(f"🔹 Analyzing {col} Prices")

            series = data[col].dropna().astype(float)
            x = np.arange(len(series))

            # Step 1: KPSS test
            kpss_p = kpss_test(series)
            st.write(f"KPSS p-value: **{kpss_p:.4f}**")

            # Step 2: Detrend or difference
            if kpss_p > 0.05:
                st.info("✅ Series is trend stationary — applying polynomial detrending.")
                # Convert to numpy before fit
                y = np.asarray(series.values, dtype=float)
                best_deg, best_r2, best_p = 1, -np.inf, None
                for deg in range(1, 6):
                    try:
                        p = Polynomial.fit(x, y, deg)
                        y_fit = p(x)
                        r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)
                        if r2 > best_r2:
                            best_deg, best_r2, best_p = deg, r2, p
                    except Exception as e:
                        continue
                trend = best_p(x)
                processed_series = y - trend
                d_order = 0
            else:
                st.warning("⚠️ Series is difference stationary — using differencing.")
                y = np.asarray(series.values, dtype=float)
                processed_series = np.diff(y)
                trend = np.zeros_like(y)
                d_order = 1

            progress_bar.progress(40)
            progress_text.text(f"Fitting ARIMA model for {col}...")

            # Step 3: Fit best ARIMA model
            best_aic, best_order, best_model = np.inf, None, None
            for p in range(6):
                for q in range(6):
                    try:
                        model = ARIMA(processed_series, order=(p, d_order, q)).fit()
                        if model.aic < best_aic:
                            best_aic, best_order, best_model = model.aic, (p, d_order, q), model
                    except:
                        continue

            st.write(f"✅ Best ARIMA order: **{best_order}**, AIC = {best_aic:.2f}")

            # Step 4: Diagnostics
            residuals = best_model.resid
            shapiro_stat, shapiro_p = shapiro(residuals)
            ljung_box_p = acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].values[0]

            shapiro_msg = (
                f"🟩 **Shapiro–Wilk p = {shapiro_p:.4f} → Residuals normal**"
                if shapiro_p > 0.05
                else f"🟥 **Shapiro–Wilk p = {shapiro_p:.4f} → Not normal**"
            )
            ljung_msg = (
                f"🟩 **Ljung–Box p = {ljung_box_p:.4f} → No autocorrelation**"
                if ljung_box_p > 0.05
                else f"🟥 **Ljung–Box p = {ljung_box_p:.4f} → Autocorrelation detected**"
            )
            st.markdown(shapiro_msg)
            st.markdown(ljung_msg)

            progress_bar.progress(60)
            progress_text.text(f"Generating fitted and forecast data for {col}...")

            # Step 5: Fitted vs original
            fitted = best_model.fittedvalues
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series.index, series, label="Original", color="blue")
            ax.plot(series.index[-len(fitted):], trend[-len(fitted):] + fitted, label="Fitted", color="orange")
            ax.set_title(f"{col} – Original vs Fitted")
            ax.set_ylabel(f"Price ({symbol})")
            ax.legend()
            st.pyplot(fig)

            # Step 6: Forecast back to original scale
            forecast_steps = 10
            forecast_diff = best_model.forecast(steps=forecast_steps)
            last_x = np.arange(len(x), len(x) + forecast_steps)
            poly_forecast = best_p(last_x) if kpss_p > 0.05 else np.zeros(forecast_steps)
            full_forecast = poly_forecast + forecast_diff if kpss_p > 0.05 else forecast_diff + y[-1]

            forecast_df = pd.DataFrame({
                "Polynomial Trend": poly_forecast,
                "ARIMA Component": forecast_diff,
                "Combined Forecast (Original Scale)": full_forecast
            })

            st.write(f"**{col} Forecast (Next {forecast_steps} Periods, in {symbol}):**")
            st.dataframe(forecast_df.style.format(f"{symbol}{{:.2f}}"))

            summary_rows.append({
                "Series": col,
                "ARIMA Order": best_order,
                "AIC": round(best_aic, 2),
                "Shapiro p": round(shapiro_p, 4),
                "Ljung-Box p": round(ljung_box_p, 4)
            })

        progress_bar.progress(100)
        progress_text.text("✅ Analysis completed for all series.")

        # Summary table
        st.subheader("📊 Summary of Results for All Series")
        st.dataframe(pd.DataFrame(summary_rows))

    except Exception as e:
        st.error(f"Error occurred: {e}")
