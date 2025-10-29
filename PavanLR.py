# 📈 ARIMA + Polynomial Detrending Dashboard (Fixed ARIMA Version)
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
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
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
                    for deg in range(1, 3):  # Reduced to 2 for stability
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

                # Ensure processed data is valid and not constant
                if processed is None or len(processed) < 10:
                    st.warning(f"Not enough valid detrended data for {col}, using simple differencing.")
                    processed = np.diff(y, n=1)
                    d_order = 1

                # Check if processed data has variance
                if np.std(processed) < 1e-10:
                    st.warning("Processed data has very low variance, adding small noise for stability.")
                    processed = processed + np.random.normal(0, 1e-6, len(processed))

                # --- Step 3: Fit ARIMA with more robust approach ---
                best_model, best_order, best_aic = None, None, np.inf
                
                # Try simpler orders first
                orders_to_try = [
                    (0, 0, 0),  # White noise model
                    (1, 0, 0),  # AR(1)
                    (0, 0, 1),  # MA(1)
                    (1, 0, 1),  # ARMA(1,1)
                    (0, 1, 0),  # Random walk
                    (1, 1, 0),  # ARIMA(1,1,0)
                    (0, 1, 1),  # ARIMA(0,1,1)
                    (1, 1, 1),  # ARIMA(1,1,1)
                ]
                
                successful_models = []
                
                for order in orders_to_try:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # Adjust differencing order based on our preprocessing
                            actual_d_order = d_order + order[1]
                            final_order = (order[0], actual_d_order, order[2])
                            
                            # For zero differencing, use original processed data
                            if actual_d_order == 0:
                                model_data = processed
                            else:
                                # Use original data and let ARIMA handle differencing
                                model_data = y
                                final_order = (order[0], actual_d_order, order[2])
                            
                            model = ARIMA(model_data, order=final_order).fit()
                            
                            if not np.isnan(model.aic) and np.isfinite(model.aic):
                                successful_models.append((model, final_order, model.aic))
                                
                    except Exception as e:
                        continue

                # Select best model from successful ones
                if successful_models:
                    successful_models.sort(key=lambda x: x[2])  # Sort by AIC
                    best_model, best_order, best_aic = successful_models[0]
                    st.success(f"✅ Best ARIMA order: {best_order}, AIC = {best_aic:.2f}")
                else:
                    # Last resort: simple mean model
                    st.warning("No ARIMA models converged, using simple forecasting")
                    class SimpleModel:
                        def __init__(self, data):
                            self.data = data
                            self.resid = np.diff(data) - np.mean(np.diff(data))
                            self.aic = 1000  # Placeholder
                        
                        def forecast(self, steps):
                            last_value = self.data[-1]
                            drift = np.mean(np.diff(self.data[-10:])) if len(self.data) > 10 else 0
                            return np.array([last_value + drift * (i+1) for i in range(steps)])
                        
                        @property
                        def fittedvalues(self):
                            return self.data[:-1]  # Simple placeholder
                    
                    best_model = SimpleModel(y)
                    best_order = (0, 1, 0)
                    best_aic = 1000

                # --- Step 4: Diagnostics (only if we have a proper model) ---
                if hasattr(best_model, 'resid'):
                    residuals = best_model.resid
                    residuals_clean = residuals[~np.isnan(residuals)]
                    
                    if len(residuals_clean) > 3:
                        try:
                            shapiro_p = shapiro(residuals_clean)[1]
                        except:
                            shapiro_p = 0.0
                    else:
                        shapiro_p = 0.0
                        
                    if len(residuals_clean) > 5:
                        try:
                            ljung_p = acorr_ljungbox(residuals_clean, lags=[min(5, len(residuals_clean)-1)], return_df=True)["lb_pvalue"].iloc[0]
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
                else:
                    shapiro_p = 0.0
                    ljung_p = 0.0
                    st.info("Simple model used - diagnostics skipped")

                # --- Step 5: Forecasting ---
                try:
                    forecast_values = best_model.forecast(steps=forecast_steps)
                    # Ensure forecast is properly shaped
                    if hasattr(forecast_values, 'values'):
                        forecast_values = forecast_values.values
                    forecast_values = np.array(forecast_values).flatten()
                except:
                    # Fallback forecasting
                    last_value = y[-1]
                    trend = np.mean(np.diff(y[-10:])) if len(y) > 10 else 0
                    forecast_values = np.array([last_value + trend * (i+1) for i in range(forecast_steps)])

                forecast_df = pd.DataFrame({
                    "Step": range(1, forecast_steps + 1),
                    "Forecast": forecast_values
                })

                # --- Step 6: Plot ---
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(series.index, y, label="Original", color="blue", linewidth=1)
                
                # Plot fitted values if available
                if hasattr(best_model, 'fittedvalues') and len(best_model.fittedvalues) > 0:
                    fitted_len = len(best_model.fittedvalues)
                    if fitted_len <= len(y):
                        ax.plot(series.index[-fitted_len:], 
                               y[-fitted_len:] if d_order == 1 else best_model.fittedvalues,
                               label="Fitted", color="orange", linewidth=1)
                
                # Plot forecast
                future_dates = pd.date_range(start=series.index[-1], periods=forecast_steps+1, freq='D')[1:]
                ax.plot(future_dates, forecast_values, label="Forecast", color="red", linestyle="--", linewidth=1)
                
                ax.set_title(f"{col} — Price Analysis & Forecast")
                ax.set_ylabel(f"Price ({symbol})")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

                # Print forecast values
                st.write(f"**Forecasted {col} Values ({symbol}):**")
                forecast_display = forecast_df.copy()
                forecast_display["Forecast"] = forecast_display["Forecast"].round(2)
                st.dataframe(forecast_display.style.format({"Forecast": "{:.2f}"}))

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
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df)
            
            # Show best model
            best_model_row = summary_df.loc[summary_df['AIC'].idxmin()]
            st.success(f"**Best performing model:** {best_model_row['Series']} with ARIMA{best_model_row['ARIMA']} (AIC: {best_model_row['AIC']})")
        else:
            st.warning("No successful analyses completed.")

        progress_bar.progress(100)
        progress_text.text("Analysis completed!")

    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
        st.info("Try using a different ticker or date range. Recommended: AAPL, MSFT, GOOGL")
