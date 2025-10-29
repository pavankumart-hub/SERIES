# 📊 Advanced ARIMA Stationarity & Diagnostics Dashboard
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

st.set_page_config(page_title="Advanced ARIMA Dashboard", layout="wide")

# --- Title ---
st.title("📈 Advanced ARIMA Stationarity & Diagnostics Dashboard")

# --- Inputs ---
ticker = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS or AAPL):", "RELIANCE.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
fit_button = st.button("Run Full ARIMA Analysis")

# --- Function: KPSS Test ---
def kpss_test(series):
    statistic, p_value, _, _ = kpss(series, regression="c", nlags="auto")
    return p_value

# --- Function: Progress Bar ---
def progress(percent, text):
    st.progress(percent)
    st.write(f"**{text} ({percent*100:.0f}% completed)**")

if fit_button:
    progress(0.1, "Downloading data...")

    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found. Check symbol or date range.")
    else:
        progress(0.2, "Data downloaded successfully.")

        # --- Basic Info ---
        st.subheader("📋 Company Information")
        st.write(f"**Ticker:** {ticker}")
        st.write(f"**Number of data points collected:** {len(data)}")
        st.write(f"**Date Range:** {start_date} to {end_date}")

        # --- Plot All Price Series ---
        st.subheader("📊 Price Series Overview")
        fig, ax = plt.subplots(figsize=(10, 4))
        data[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
        ax.set_title("All Price Series (Open, High, Low, Close)")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        symbol = "₹" if ticker.endswith(".NS") else "$"

        progress(0.3, "Starting ARIMA model fitting for each series...")

        # --- Loop through each column ---
        results_summary = []

        for col in ['Open', 'High', 'Low', 'Close']:
            st.subheader(f"🔹 Analyzing {col} Prices")
            series = data[col].dropna()

            # --- Step 1: KPSS Test ---
            kpss_p = kpss_test(series)
            st.write(f"KPSS p-value: **{kpss_p:.4f}**")

            # --- Step 2: Detrending or Differencing ---
            if kpss_p > 0.05:
                st.success("✅ Series is trend stationary — fitting polynomial trend (<10)...")
                x = np.arange(len(series))
                best_deg, best_r2, best_p = 1, -np.inf, None
                for deg in range(1, 10):
                    p = Polynomial.fit(x, series, deg)
                    y_fit = p(x)
                    r2 = 1 - np.sum((series - y_fit) ** 2) / np.sum((series - np.mean(series)) ** 2)
                    if r2 > best_r2:
                        best_deg, best_r2, best_p = deg, r2, p
                trend = best_p(x)
                processed_series = series - trend
                d_order = 0
            else:
                st.warning("⚠️ Series is difference stationary — applying first differencing.")
                processed_series = series.diff().dropna()
                trend = np.zeros_like(series)
                d_order = 1

            progress(0.5, f"Processed {col} series for stationarity...")

            # --- Step 3: Fit Best ARIMA ---
            best_aic, best_order, best_model = np.inf, None, None
            for p in range(6):
                for q in range(6):
                    try:
                        model = ARIMA(processed_series, order=(p, d_order, q)).fit()
                        if model.aic < best_aic:
                            best_aic, best_order, best_model = model.aic, (p, d_order, q), model
                    except:
                        continue

            st.write(f"✅ Best ARIMA order for {col}: **{best_order}**, AIC = {best_aic:.2f}")

            # --- Step 4: Residuals ---
            residuals = best_model.resid
            fig, ax = plt.subplots(2, 1, figsize=(10, 5))
            ax[0].plot(residuals)
            ax[0].set_title(f"{col} Residuals Over Time")
            ax[1].hist(residuals, bins=30, alpha=0.7)
            ax[1].set_title(f"{col} Residual Histogram")
            st.pyplot(fig)

            # --- Step 5: Normality & Independence ---
            shapiro_stat, shapiro_p = shapiro(residuals)
            ljung_box_p = acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].values[0]

            if shapiro_p > 0.05:
                st.markdown(f"**🟩 Shapiro–Wilk Normality Test p = {shapiro_p:.4f} → Residuals are normal**")
            else:
                st.markdown(f"**🟥 Shapiro–Wilk Normality Test p = {shapiro_p:.4f} → Not normal**")

            if ljung_box_p > 0.05:
                st.markdown(f"**🟩 Ljung–Box Test p = {ljung_box_p:.4f} → No autocorrelation**")
            else:
                st.markdown(f"**🟥 Ljung–Box Test p = {ljung_box_p:.4f} → Autocorrelation detected**")

            progress(0.7, f"Fitted ARIMA model for {col}...")

            # --- Step 6: Overlay Original vs Fitted ---
            fitted = best_model.fittedvalues
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series.index, series, label="Original", color="blue")
            ax.plot(series.index[-len(fitted):], fitted, label="Fitted", color="orange")
            ax.set_title(f"{col} – Original vs Fitted")
            ax.legend()
            st.pyplot(fig)

            # --- Step 7: Forecast (No Plot) ---
            forecast_values = best_model.forecast(steps=10)
            forecast_df = pd.DataFrame({"Forecast": forecast_values})
            st.write(f"**{col} Forecast (Next 10 Periods):**")
            st.dataframe(forecast_df.style.format(f"{symbol}{{:.2f}}"))

            results_summary.append({
                "Series": col,
                "ARIMA Order": best_order,
                "AIC": round(best_aic, 2),
                "Shapiro p": round(shapiro_p, 4),
                "Ljung-Box p": round(ljung_box_p, 4)
            })

        progress(1.0, "✅ Full ARIMA analysis completed for all series.")

        # --- Summary Table ---
        st.subheader("📊 Summary of Results for All Series")
        st.dataframe(pd.DataFrame(results_summary))
