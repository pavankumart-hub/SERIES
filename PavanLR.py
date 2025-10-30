# 📈 ARIMA + Polynomial Detrending Dashboard (Fixed KPSS Version)
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

# FIXED KPSS helper function
def kpss_test(series):
    """
    Perform KPSS test for stationarity
    Returns: p-value (float)
    """
    try:
        # Clean the series - remove NaNs and ensure it's numeric
        series_clean = series.dropna()
        if len(series_clean) < 10:  # Need minimum data points
            return 0.0
            
        # Perform KPSS test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value, n_lags, critical_values = kpss(
                series_clean, 
                regression='c',  # constant trend
                nlags='auto'     # automatic lag selection
            )
        return p_value
    except Exception as e:
        st.warning(f"KPSS test failed: {str(e)}")
        return 0.0  # Default to non-stationary if test fails

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

                # --- Step 1: KPSS Test (FIXED) ---
                st.write("**🔍 Performing KPSS Stationarity Test...**")
                kpss_p = kpss_test(series)
                
                # Display KPSS results clearly
                st.write(f"**KPSS Test p-value:** {kpss_p:.6f}")
                
                if kpss_p > 0.05:
                    st.success("✅ **KPSS Result:** Series is TREND STATIONARY (p > 0.05)")
                    st.info("**Action:** Applying polynomial detrending")
                else:
                    st.warning("⚠️ **KPSS Result:** Series is DIFFERENCE STATIONARY (p ≤ 0.05)")
                    st.info("**Action:** Applying differencing")

                # --- Step 2: Detrending or differencing ---
                best_poly, trend, processed = None, np.zeros_like(y), None
                d_order = 0
                
                if kpss_p > 0.05:
                    # Trend stationary - use polynomial detrending
                    best_deg, best_r2 = 1, -np.inf
                    for deg in range(1, 3):
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
                        st.write(f"**Polynomial Detrending:** Degree {best_deg}, R² = {best_r2:.4f}")
                        
                        # Plot detrending result
                        fig_detrend, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Original + trend
                        ax1.plot(series.index, y, label='Original', color='blue')
                        ax1.plot(series.index, trend, label=f'Poly Trend (deg {best_deg})', color='red')
                        ax1.set_title(f'{col} - Original Series with Polynomial Trend')
                        ax1.legend()
                        
                        # Detrended series
                        ax2.plot(series.index, processed, label='Detrended', color='green')
                        ax2.set_title(f'{col} - Detrended Series')
                        ax2.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig_detrend)
                        plt.close()
                    else:
                        st.warning("⚠️ Polynomial detrending failed, using differencing.")
                        processed = np.diff(y, n=1)
                        d_order = 1
                else:
                    # Difference stationary - use differencing
                    processed = np.diff(y, n=1)
                    d_order = 1
                    st.write(f"**Differencing Applied:** Order {d_order}")

                # Ensure processed data is valid
                if processed is None or len(processed) < 10:
                    st.warning(f"Not enough valid processed data for {col}, using simple differencing.")
                    processed = np.diff(y, n=1)
                    d_order = 1

                # Check if processed data has variance
                if np.std(processed) < 1e-10:
                    st.warning("Processed data has very low variance, adding small noise for stability.")
                    processed = processed + np.random.normal(0, 1e-6, len(processed))

                # --- Step 3: Fit ARIMA ---
                best_model, best_order, best_aic = None, None, np.inf
                
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
                            actual_d_order = d_order + order[1]
                            final_order = (order[0], actual_d_order, order[2])
                            
                            if actual_d_order == 0:
                                model_data = processed
                            else:
                                model_data = y
                                final_order = (order[0], actual_d_order, order[2])
                            
                            model = ARIMA(model_data, order=final_order).fit()
                            
                            if not np.isnan(model.aic) and np.isfinite(model.aic):
                                successful_models.append((model, final_order, model.aic))
                                
                    except Exception as e:
                        continue

                # Select best model
                if successful_models:
                    successful_models.sort(key=lambda x: x[2])
                    best_model, best_order, best_aic = successful_models[0]
                    st.success(f"✅ **Best ARIMA Model:** {best_order}, AIC = {best_aic:.2f}")
                else:
                    st.warning("No ARIMA models converged, using simple forecasting")
                    class SimpleModel:
                        def __init__(self, data):
                            self.data = data
                            self.resid = np.diff(data) - np.mean(np.diff(data))
                            self.aic = 1000
                        
                        def forecast(self, steps):
                            last_value = self.data[-1]
                            drift = np.mean(np.diff(self.data[-10:])) if len(self.data) > 10 else 0
                            return np.array([last_value + drift * (i+1) for i in range(steps)])
                        
                        @property
                        def fittedvalues(self):
                            return self.data[:-1]
                    
                    best_model = SimpleModel(y)
                    best_order = (0, 1, 0)
                    best_aic = 1000

                # --- Step 4: Diagnostics ---
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

                    # Display diagnostic results
                    st.subheader("📊 Model Diagnostics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if shapiro_p > 0.05:
                            st.success(f"✅ **Normality Test:** p = {shapiro_p:.4f} (Normal)")
                        else:
                            st.error(f"❌ **Normality Test:** p = {shapiro_p:.4f} (Not Normal)")
                    
                    with col2:
                        if ljung_p > 0.05:
                            st.success(f"✅ **Autocorrelation Test:** p = {ljung_p:.4f} (No AC)")
                        else:
                            st.error(f"❌ **Autocorrelation Test:** p = {ljung_p:.4f} (AC Detected)")
                else:
                    shapiro_p = 0.0
                    ljung_p = 0.0
                    st.info("Simple model used - diagnostics skipped")

                # --- Step 5: Forecasting ---
                try:
                    forecast_values = best_model.forecast(steps=forecast_steps)
                    if hasattr(forecast_values, 'values'):
                        forecast_values = forecast_values.values
                    forecast_values = np.array(forecast_values).flatten()
                except:
                    last_value = y[-1]
                    trend = np.mean(np.diff(y[-10:])) if len(y) > 10 else 0
                    forecast_values = np.array([last_value + trend * (i+1) for i in range(forecast_steps)])

                forecast_df = pd.DataFrame({
                    "Step": range(1, forecast_steps + 1),
                    "Forecast": forecast_values
                })

                # --- Step 6: Plot Results ---
                st.subheader("📈 Forecast Results")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(series.index, y, label="Original", color="blue", linewidth=1)
                
                if hasattr(best_model, 'fittedvalues') and len(best_model.fittedvalues) > 0:
                    fitted_len = len(best_model.fittedvalues)
                    if fitted_len <= len(y):
                        ax.plot(series.index[-fitted_len:], 
                               y[-fitted_len:] if d_order == 1 else best_model.fittedvalues,
                               label="Fitted", color="orange", linewidth=1)
                
                future_dates = pd.date_range(start=series.index[-1], periods=forecast_steps+1, freq='D')[1:]
                ax.plot(future_dates, forecast_values, label="Forecast", color="red", linestyle="--", linewidth=1)
                
                ax.set_title(f"{col} — ARIMA{best_order} Forecast")
                ax.set_ylabel(f"Price ({symbol})")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

                # Display forecast values
                st.write(f"**Forecasted {col} Prices:**")
                forecast_display = forecast_df.copy()
                forecast_display["Forecast"] = forecast_display["Forecast"].round(2)
                st.dataframe(forecast_display.style.format({"Forecast": "{:.2f}"}))

                results_summary.append({
                    "Series": col,
                    "KPSS p-value": round(kpss_p, 4),
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
            
            best_model_row = summary_df.loc[summary_df['AIC'].idxmin()]
            st.success(f"**Best performing model:** {best_model_row['Series']} with ARIMA{best_model_row['ARIMA']} (AIC: {best_model_row['AIC']})")
        else:
            st.warning("No successful analyses completed.")

        progress_bar.progress(100)
        progress_text.text("Analysis completed!")

    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
        st.info("Try using a different ticker or date range. Recommended: AAPL, MSFT, GOOGL")
