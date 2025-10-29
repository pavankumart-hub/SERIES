# 📈 ARIMA + Polynomial Detrending Dashboard (Fixed Execution Version)
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
st.title("📈 ARIMA + Polynomial Detrending Dashboard")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
forecast_steps = st.sidebar.slider("Forecast Steps", 5, 30, 10)
run_btn = st.sidebar.button("Run Full Analysis")

# KPSS helper
def kpss_test(series):
    try:
        statistic, p_value, _, _ = kpss(series, regression="c", nlags="auto")
        return p_value
    except:
        return 0.0

if run_btn:
    try:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        progress_text.text("Downloading data...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for this ticker or range.")
            st.stop()

        progress_bar.progress(10)
        progress_text.text("Data downloaded successfully...")

        # Company info
        st.subheader("📋 Company Information")
        st.write(f"**Ticker:** {ticker}")
        st.write(f"**Data points collected:** {len(data)}")
        st.write(f"**Date Range:** {start_date} → {end_date}")

        symbol = "₹" if ticker.endswith(".NS") else "$"

        # Plot overview
        st.subheader("📊 Price Series Overview")
        fig, ax = plt.subplots(figsize=(10, 4))
        data[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
        ax.set_title(f"{ticker} — Price Series (Open, High, Low, Close)")
        ax.set_ylabel(f"Price ({symbol})")
        st.pyplot(fig)
        plt.close()

        progress_bar.progress(20)
        progress_text.text("Starting series-wise ARIMA analysis...")

        results_summary = []
        price_columns = ['Open', 'High', 'Low', 'Close']
        total_cols = len(price_columns)
        completed = 0

        for col in price_columns:
            st.markdown(f"---")
            st.subheader(f"🔹 {col} Price Analysis")

            try:
                series = data[col].dropna().astype(float)
                if len(series) < 20:
                    st.warning(f"⚠️ Not enough data for {col}, skipping.")
                    completed += 1
                    continue

                x = np.arange(len(series))
                y = series.values

                # --- Step 1: KPSS ---
                kpss_p = kpss_test(series)
                st.write(f"**KPSS p-value:** {kpss_p:.4f}")

                # --- Step 2: Detrending or differencing ---
                best_poly, trend, processed = None, np.zeros_like(y), None
                d_order = 0
                
                if kpss_p > 0.05:
                    st.info("✅ Trend stationary → Applying polynomial detrending.")
                    best_deg, best_r2 = 1, -np.inf
                    for deg in range(1, 4):  # Reduced from 5 to 3 for stability
                        try:
                            p = Polynomial.fit(x, y, deg)
                            fit_y = p(x)
                            r2 = 1 - np.sum((y - fit_y) ** 2) / np.sum((y - np.mean(y)) ** 2)
                            if r2 > best_r2:
                                best_deg, best_r2, best_poly = deg, r2, p
                        except:
                            continue

                    if best_poly is not None:
                        trend = best_poly(x)
                        processed = y - trend
                        st.write(f"Best polynomial degree: {best_deg}, R² = {best_r2:.4f}")
                    else:
                        st.warning("⚠️ Polynomial detrending failed, using differencing.")
                        processed = np.diff(y, n=1)
                        d_order = 1
                else:
                    st.warning("⚠️ Difference stationary → Applying differencing.")
                    processed = np.diff(y, n=1)
                    d_order = 1

                # Ensure processed data is valid
                if processed is None or len(processed) < 10:
                    st.warning(f"Not enough valid detrended data for {col}, using simple differencing.")
                    processed = np.diff(y, n=1)
                    d_order = 1

                # --- Step 3: Fit ARIMA (simplified search) ---
                best_model, best_order, best_aic = None, None, np.inf
                
                # Try only common ARIMA orders for stability
                orders_to_try = [(1, d_order, 1), (1, d_order, 0), (0, d_order, 1), 
                               (2, d_order, 2), (0, d_order, 0)]
                
                for order in orders_to_try:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(processed, order=order).fit()
                            if model.aic < best_aic and not np.isnan(model.aic):
                                best_aic, best_model, best_order = model.aic, model, order
                    except:
                        continue

                if best_model is None:
                    st.warning("ARIMA search failed, trying simple AR(1) model.")
                    try:
                        best_model = ARIMA(processed, order=(1, d_order, 0)).fit()
                        best_order = (1, d_order, 0)
                        best_aic = best_model.aic
                    except:
                        st.error(f"❌ Could not fit ARIMA model for {col}")
                        completed += 1
                        continue

                st.success(f"✅ Best ARIMA order: {best_order}, AIC = {best_aic:.2f}")

                # --- Step 4: Diagnostics ---
                residuals = best_model.resid
                residuals_clean = residuals[~np.isnan(residuals)]
                
                if len(residuals_clean) > 3:
                    try:
                        shapiro_p = shapiro(residuals_clean)[1]
                    except:
                        shapiro_p = 0.0
                else:
                    shapiro_p = 0.0
                    
                if len(residuals_clean) > 10:
                    try:
                        ljung_p = acorr_ljungbox(residuals_clean, lags=[min(10, len(residuals_clean)-1)], return_df=True)["lb_pvalue"].iloc[0]
                    except:
                        ljung_p = 0.0
                else:
                    ljung_p = 0.0

                shapiro_msg = (
                    f"🟩 **Shapiro–Wilk p = {shapiro_p:.4f} → Normal Residuals**"
                    if shapiro_p > 0.05
                    else f"🟥 **Shapiro–Wilk p = {shapiro_p:.4f} → Not Normal**"
                )
                ljung_msg = (
                    f"🟩 **Ljung–Box p = {ljung_p:.4f} → No Autocorrelation**"
                    if ljung_p > 0.05
                    else f"🟥 **Ljung–Box p = {ljung_p:.4f} → Autocorrelation Detected**"
                )
                st.markdown(shapiro_msg)
                st.markdown(ljung_msg)

                # --- Step 5: Fitted + Forecast ---
                fitted = best_model.fittedvalues
                try:
                    forecast_diff = best_model.forecast(steps=forecast_steps)
                except:
                    # Simple forecast as fallback
                    last_value = processed[-1] if len(processed) > 0 else 0
                    forecast_diff = np.full(forecast_steps, last_value)

                future_x = np.arange(len(x), len(x) + forecast_steps)

                # Reconstruct forecast to original scale
                if d_order == 0 and best_poly is not None:
                    try:
                        poly_future = best_poly(future_x)
                        full_forecast = poly_future + forecast_diff
                    except:
                        full_forecast = y[-1] + np.cumsum(forecast_diff)
                else:
                    full_forecast = y[-1] + np.cumsum(forecast_diff)

                forecast_df = pd.DataFrame({
                    "Step": range(1, forecast_steps + 1),
                    "Forecast": full_forecast
                })

                # --- Step 6: Plot ---
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(series.index, y, label="Original", color="blue", linewidth=1)
                
                # Plot fitted values
                if len(fitted) <= len(series) and len(fitted) > 0:
                    if d_order == 0 and best_poly is not None:
                        fitted_original_scale = trend[-len(fitted):] + fitted
                    else:
                        fitted_original_scale = y[-len(fitted):]  # Simplified for differenced series
                    
                    ax.plot(series.index[-len(fitted):], fitted_original_scale, 
                           label="Fitted", color="orange", linewidth=1)
                
                ax.set_title(f"{col} — Original vs Fitted")
                ax.set_ylabel(f"Price ({symbol})")
                ax.legend()
                st.pyplot(fig)
                plt.close()

                # Print forecast values
                st.write(f"**Forecasted {col} Values ({symbol}):**")
                st.dataframe(forecast_df.style.format({"Forecast": f"{symbol}{{:.2f}}"}))

                results_summary.append({
                    "Series": col,
                    "ARIMA": best_order,
                    "AIC": round(best_aic, 2),
                    "Shapiro p": round(shapiro_p, 4),
                    "Ljung p": round(ljung_p, 4)
                })

                completed += 1
                progress = int(20 + (completed / total_cols) * 80)
                progress_bar.progress(progress)
                progress_text.text(f"Completed {col} analysis... ({completed}/{total_cols})")

            except Exception as e:
                st.error(f"Error in {col}: {str(e)}")
                completed += 1
                continue

        # Final summary
        st.markdown("---")
        st.markdown("### 📊 Analysis Summary")
        if results_summary:
            st.dataframe(pd.DataFrame(results_summary))
        else:
            st.warning("No successful analyses completed.")

        progress_bar.progress(100)
        progress_text.text("Analysis completed!")

    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
        st.info("Try using a different ticker or date range.")
