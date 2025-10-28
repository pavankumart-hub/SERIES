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
st.title("📈 Auto ARIMA Forecast with KPSS, ADF, Detrending & Diagnostics")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL)", "RELIANCE.NS").upper()
price_type = st.sidebar.selectbox("Select Price Type", ["Open", "High", "Low", "Close"])
start_date = st.sidebar.date_input("Select Start Date", datetime(2023, 1, 1))
forecast_steps = st.sidebar.slider("Forecast Steps (days)", 1, 30, 5)

# Run button
run_analysis = st.sidebar.button("🚀 Run Analysis")

# ---------------- Start when button clicked ----------------
if run_analysis:
    # ✅ 1. Download Data
    st.subheader(f"1️⃣ Downloading {price_type} data for {ticker}")
    end_date = datetime.now()

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if data.empty:
        st.error("❌ No data found for this ticker. Try adding '.NS' or '.BO' for Indian stocks.")
        st.stop()

    series = data[price_type].dropna()
    st.line_chart(series, use_container_width=True)
    st.success(f"Downloaded {len(series)} data points from {start_date} to {end_date.date()}.")

    # ✅ 2. KPSS Test
    st.subheader("2️⃣ KPSS Stationarity Test")

    def kpss_test(ts):
        stat, p, lags, crit = kpss(ts, regression="c", nlags="auto")
        return stat, p

    stat, pval = kpss_test(series)
    st.write(f"**KPSS Statistic:** {stat:.4f}, **p-value:** {pval:.4f}")

    if pval > 0.05:
        st.success("✅ Trend stationary — proceeding with detrending.")
        stationary_type = "trend_stationary"
    else:
        st.warning("⚠️ Difference stationary — applying differencing.")
        stationary_type = "difference_stationary"

    # ✅ 3. Detrending or Differencing
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
        ax.legend()
        ax.set_title("Trend Fit")
        st.pyplot(fig)

        # ✅ ADF Test on Detrended Series
        st.subheader("📊 ADF Test on Detrended Data")
        adf_stat, adf_p, _, _, _, _ = adfuller(detrended)
        st.write(f"ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.4f}")

        if adf_p < 0.05:
            st.success("✅ Detrended series is stationary — fitting ARIMA.")
            processed_series = detrended
            d = 0
        else:
            st.warning("⚠️ Detrended still non-stationary — applying differencing.")
            processed_series = detrended.diff().dropna()
            d = 1

    else:
        st.subheader("3️⃣ Applying First-Order Differencing")
        processed_series = series.diff().dropna()
        st.line_chart(processed_series)
        st.info("Performed differencing to achieve stationarity.")
        d = 1

    # ✅ 4. ARIMA Grid Search
    st.subheader("4️⃣ Automatic ARIMA Model Selection (p,q ∈ [0,5])")
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
        st.error("❌ ARIMA fitting failed for all combinations.")
        st.stop()

    st.success(f"✅ Best ARIMA Order: {best_order}, AIC={best_aic:.2f}")

    # ✅ 5. Residual Analysis
    st.subheader("5️⃣ Residual Analysis")
    residuals = best_model.resid

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(residuals, color='blue')
    ax[0].set_title("Residuals Over Time")
    ax[1].hist(residuals, bins=25, color='gray', edgecolor='black')
    ax[1].set_title("Residual Histogram")
    plt.tight_layout()
    st.pyplot(fig)

    # ✅ 6. Diagnostic Tests
    st.subheader("6️⃣ Residual Diagnostic Tests")

    shapiro_stat, shapiro_p = shapiro(residuals)
    ljung = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_p = ljung['lb_pvalue'].iloc[0]

    st.write(f"**Shapiro–Wilk p-value:** {shapiro_p:.4f}")
    st.write(f"**Ljung–Box p-value:** {lb_p:.4f}")

    if shapiro_p > 0.05:
        st.success("✅ Residuals appear normally distributed.")
    else:
        st.warning("⚠️ Residuals may not be normal.")

    if lb_p > 0.05:
        st.success("✅ No significant autocorrelation detected.")
    else:
        st.warning("⚠️ Residuals show autocorrelation.")

    # ✅ 7. Forecasting (Original Scale)
    st.subheader("7️⃣ Forecasting (Original Scale)")

    try:
        forecast = best_model.get_forecast(steps=forecast_steps)
        forecast_mean = np.array(forecast.predicted_mean)
        conf_int = np.array(forecast.conf_int())

        # Generate forecast index
        forecast_index = pd.date_range(series.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
        
        # Ensure forecast_index has the same length as forecast
        if len(forecast_index) != len(forecast_mean):
            forecast_index = forecast_index[:len(forecast_mean)]

        # Reconstruct to original scale
        if stationary_type == "trend_stationary":
            # For trend stationary: add back the trend component
            future_x = np.arange(len(series), len(series) + len(forecast_mean)).reshape(-1, 1)
            trend_future = best_model.predict(best_poly.transform(future_x))
            
            # Ensure arrays have same length
            min_len = min(len(forecast_mean), len(trend_future))
            forecast_mean = forecast_mean[:min_len]
            trend_future = trend_future[:min_len]
            conf_int = conf_int[:min_len]
            
            forecast_mean_orig = forecast_mean + trend_future
            conf_int_orig = conf_int + trend_future.reshape(-1, 1)
            
        else:
            # For difference stationary: cumulative sum + last value
            forecast_mean_orig = np.cumsum(forecast_mean) + series.iloc[-1]
            conf_int_orig = conf_int + series.iloc[-1]

        # Final alignment check
        min_len = min(len(forecast_index), len(forecast_mean_orig))
        forecast_index = forecast_index[:min_len]
        forecast_mean_orig = forecast_mean_orig[:min_len]
        conf_int_orig = conf_int_orig[:min_len]

        # Plotting with safety checks
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(series.index, series, label="Historical", color="blue", linewidth=2)
        
        # Plot forecast
        ax.plot(forecast_index, forecast_mean_orig, label="Forecast", color="red", linewidth=2, marker='o')
        
        # Plot confidence interval
        ax.fill_between(forecast_index, conf_int_orig[:, 0], conf_int_orig[:, 1], 
                       color="pink", alpha=0.3, label="95% CI")
        
        ax.legend()
        ax.set_title(f"{ticker} {price_type} Forecast (ARIMA{best_order})")
        ax.set_xlabel("Date")
        ax.set_ylabel(price_type)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Forecast table
        forecast_df = pd.DataFrame({
            "Date": forecast_index,
            f"Forecasted {price_type}": forecast_mean_orig,
            "Lower CI": conf_int_orig[:, 0],
            "Upper CI": conf_int_orig[:, 1]
        })
        
        st.write("### 📅 Forecasted Values (Original Scale)")
        st.dataframe(forecast_df.style.format({
            f"Forecasted {price_type}": "{:.2f}",
            "Lower CI": "{:.2f}", 
            "Upper CI": "{:.2f}"
        }))

    except Exception as e:
        st.error(f"❌ Error in forecasting: {str(e)}")
        st.info("Try reducing forecast steps or using a different stock.")

    st.markdown("---")
    st.markdown("Built with ❤️ | KPSS • ADF • ARIMA • Shapiro–Wilk • Ljung–Box • Forecast (Original Scale)")

else:
    st.info("👈 Enter inputs and click **Run Analysis** to begin.")
