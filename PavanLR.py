# 📈 ARIMA Stationarity & Forecast App (Fixed Version)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from numpy.polynomial import Polynomial
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Trend & Stationarity", layout="wide")

# --- Title ---
st.title("📈 ARIMA Model with KPSS Stationarity, Detrending & Tests")

# --- Input ---
ticker = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS or AAPL):", "RELIANCE.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
fit_button = st.button("Run Full ARIMA Analysis")

if fit_button:
    with st.spinner("Downloading data..."):
        data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found. Check the symbol or date range.")
    else:
        st.subheader("1️⃣ Price Series")
        st.line_chart(data["High"], use_container_width=True)

        series = data["High"].dropna()

        # --- KPSS Test ---
        def kpss_test(series):
            statistic, p_value, _, _ = kpss(series, regression="c", nlags="auto")
            return p_value

        kpss_p = kpss_test(series)
        st.write(f"KPSS p-value: **{kpss_p:.4f}**")

        # --- Stationarity Decision ---
        if kpss_p > 0.05:
            st.success("✅ Series appears trend stationary. Fitting polynomial trend (degree < 10)...")
            best_deg, best_r2 = 1, -np.inf
            x = np.arange(len(series))
            for deg in range(1, 10):
                p = Polynomial.fit(x, series, deg)
                y_fit = p(x)
                r2 = 1 - np.sum((series - y_fit) ** 2) / np.sum((series - np.mean(series)) ** 2)
                if r2 > best_r2:
                    best_deg, best_r2, best_p = deg, r2, p
            trend = best_p(x)
            detrended = series - trend
            processed_series = detrended
            st.write(f"Best polynomial degree: **{best_deg}**")
            st.line_chart(pd.DataFrame({"Actual": series, "Fitted Trend": trend}), use_container_width=True)
        else:
            st.warning("⚠️ Series is difference stationary. Applying first differencing...")
            processed_series = series.diff().dropna()

        # --- ACF, PACF and ARIMA fitting ---
        st.subheader("2️⃣ ARIMA Model Fitting")
        best_aic, best_order, best_model = np.inf, None, None

        for p in range(6):
            for q in range(6):
                try:
                    model = ARIMA(processed_series, order=(p, 0 if kpss_p > 0.05 else 1, q)).fit()
                    if model.aic < best_aic:
                        best_aic, best_order, best_model = model.aic, (p, 0 if kpss_p > 0.05 else 1, q), model
                except:
                    continue

        st.write(f"✅ Best ARIMA order selected: **{best_order}** with AIC = {best_aic:.2f}")

        # --- Residual Analysis ---
        residuals = best_model.resid
        st.subheader("3️⃣ Residual Analysis")

        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(residuals)
        ax[0].set_title("Residuals Over Time")
        ax[1].hist(residuals, bins=30, alpha=0.7)
        ax[1].set_title("Residual Histogram")
        st.pyplot(fig)

        # --- Normality & Independence Tests ---
        shapiro_stat, shapiro_p = shapiro(residuals)
        ljung_box_p = acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].values[0]

        st.write(f"Shapiro-Wilk Test p-value: **{shapiro_p:.4f}**")
        st.write(f"Ljung–Box Test p-value: **{ljung_box_p:.4f}**")

        # --- Forecast ---
        st.subheader("4️⃣ Forecast (Next 10 Periods)")
        forecast_values = best_model.forecast(steps=10)
        symbol = "₹" if ticker.endswith(".NS") else "$"
        forecast_df = pd.DataFrame({
            "Forecast": forecast_values
        })
        st.dataframe(forecast_df.style.format(f"{symbol} {{:.2f}}"))

        st.success("✅ Forecasting and model analysis completed successfully.")
