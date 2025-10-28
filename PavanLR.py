import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Auto ARIMA Forecast", page_icon="📈", layout="wide")
st.title("📈 Automatic ARIMA Model with KPSS, ADF, Detrending & Diagnostics")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
price_type = st.sidebar.selectbox("Select Price Type", ["Open", "High", "Low", "Close"])
start_date = st.sidebar.date_input("Select Start Date", datetime(2023, 1, 1))
forecast_steps = st.sidebar.slider("Forecast Steps (days)", 1, 30, 5)

# Run button
run_analysis = st.sidebar.button("🚀 Run Analysis")

# ---------------- Start when button clicked ----------------
if run_analysis:
    # ---------------- Download Data ----------------
    st.subheader(f"1️⃣ Downloading {price_type} Data for {ticker}")
    end_date = datetime.now()
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        st.error("No data found for this ticker or date range.")
        st.stop()

    series = data[price_type].dropna()
    st.line_chart(series, use_container_width=True)
    st.success(f"Downloaded {len(series)} data points from {start_date} to {end_date.date()}.")

    # ---------------- KPSS Test ----------------
    st.subheader("2️⃣ KPSS Stationarity Test")

    def kpss_test(ts):
        stat, p, lags, crit = kpss(ts, regression="c", nlags="auto")
        return stat, p, crit

    stat, pval, crit = kpss_test(series)
    st.write(f"**KPSS Statistic:** {stat:.4f}, **p-value:** {pval:.4f}")

    if pval > 0.05:
        st.success("✅ Fail to reject H₀ → Series is trend stationary. Proceeding with detrending.")
        stationary_type = "trend_stationary"
    else:
        st.warning("⚠️ Reject H₀ → Series is difference stationary. Proceeding with differencing.")
        stationary_type = "difference_stationary"

    # ---------------- Detrending or Differencing ----------------
    if stationary_type == "trend_stationary":
        st.subheader("3️⃣ Detrending using Polynomial Regression (<10)")

        x = np.arange(len(series)).reshape(-1, 1)
        best_deg, best_r2 = 1, -np.inf
        best_model, best_poly = None, None

        for deg in range(1, 10):
            poly = PolynomialFeatures(degree=deg)
            X_poly = poly.fit_transform(x)
            model = LinearRegression().fit(X_poly, series)
            r2 = model.score(X_poly, series)
            if r2 > best_r2:
                best_deg, best_r2 = deg, r2
                best_model, best_poly = model, poly

        trend = best_model.predict(best_poly.transform(x))
        detrended = series - trend
        st.write(f"**Best Polynomial Degree:** {best_deg} (R²={best_r2:.4f})")

        fig, ax = plt.subplots()
        ax.plot(series.index, series, label="Original", color="blue")
        ax.plot(series.index, trend, label=f"Trend (deg={best_deg})", color="red")
        ax.legend(); ax.set_title("Trend Fit")
        st.pyplot(fig)

        # --- Step 3A: ADF test on detrended data ---
        st.subheader("📊 ADF Test on Detrended Series")
        adf_stat, adf_p, _, _, crit_vals, _ = adfuller(detrended)
        st.write(f"ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.4f}")
        if adf_p < 0.05:
            st.success("✅ Detrended series is stationary — proceed to ARIMA fitting.")
            processed_series = detrended
            d = 0
        else:
            st.warning("⚠️ Detrended series still non-stationary — applying differencing.")
            processed_series = detrended.diff().dropna()
            d = 1

    else:
        st.subheader("3️⃣ First-Order Differencing")
        processed_series = series.diff().dropna()
        st.line_chart(processed_series)
        st.info("Performed first-order differencing.")
        d = 1

    # ---------------- Automatic ARIMA Grid Search ----------------
    st.subheader("4️⃣ Automatic ARIMA Model Selection (p, q ∈ [0,5])")

    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, 6):
        for q in range(0, 6):
            try:
                model = ARIMA(processed_series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted
            except:
                continue

    if best_model is None:
        st.error("ARIMA fitting failed for all combinations.")
        st.stop()

    st.success(f"✅ Best ARIMA Order: {best_order}, AIC={best_aic:.2f}")

    # ---------------- Residual Analysis ----------------
    st.subheader("5️⃣ Residual Analysis")
    residuals = best_model.resid

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(residuals)
    ax[0].set_title("Residuals Over Time")
    ax[1].hist(residuals, bins=25, color='gray', edgecolor='black')
    ax[1].set_title("Residual Histogram")
    plt.tight_layout()
    st.pyplot(fig)

    # ---------------- Diagnostic Tests ----------------
    st.subheader("6️⃣ Residual Diagnostic Tests")

    shapiro_stat, shapiro_p = shapiro(residuals)
    ljung = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_p = ljung['lb_pvalue'].iloc[0]

    st.write(f"**Shapiro–Wilk p-value:** {shapiro_p:.4f}")
    st.write(f"**Ljung–Box p-value:** {lb_p:.4f}")

    if shapiro_p > 0.05:
        st.success("✅ Residuals appear normally distributed (Shapiro–Wilk).")
    else:
        st.warning("⚠️ Residuals may not be normal.")

    if lb_p > 0.05:
        st.success("✅ No significant autocorrelation detected (Ljung–Box).")
    else:
        st.warning("⚠️ Residuals show autocorrelation.")

    # ---------------- Forecasting ----------------
    st.subheader("7️⃣ Forecasting (in Original Scale)")

    forecast = best_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Convert back to original scale
    if stationary_type == "trend_stationary":
        # Add trend back
        future_x = np.arange(len(series), len(series) + forecast_steps).reshape(-1, 1)
        trend_future = best_model.predict(best_poly.transform(future_x))
        forecast_mean_orig = forecast_mean + trend_future
        conf_int_orig = conf_int + trend_future.reshape(-1, 1)
    else:
        # Cumulative sum for differenced series
        forecast_mean_orig = forecast_mean.cumsum() + series.iloc[-1]
        conf_int_orig = conf_int + series.iloc[-1]

    forecast_index = pd.date_range(series.index[-1] + timedelta(days=1), periods=forecast_steps, freq='B')

    # --- Plot forecast ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series, label="Original", color="blue")
    ax.plot(forecast_index, forecast_mean_orig, label="Forecast", color="red")
    ax.fill_between(forecast_index, conf_int_orig.iloc[:, 0], conf_int_orig.iloc[:, 1], color="pink", alpha=0.3)
    ax.legend()
    ax.set_title(f"{ticker} {price_type} Forecast (ARIMA{best_order}) - 95% CI (Original Scale)")
    st.pyplot(fig)

    # --- Display forecasted values ---
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        f"Forecasted {price_type}": forecast_mean_orig
    })
    st.write("### 📅 Forecasted Values (Original Scale)")
    st.dataframe(forecast_df)

    # ---------------- Footer ----------------
    st.markdown("---")
    st.markdown("Built with ❤️ | KPSS • ADF • ARIMA • Shapiro–Wilk • Ljung–Box • Forecast (Original Scale)")
else:
    st.info("👈 Enter inputs and click **Run Analysis** to begin.")
