# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Polynomial Stock Forecast (stable)", layout="wide")
st.title("Polynomial Regression + ARIMA Stock Forecast")
st.markdown("Center & scale date before polynomial transform to avoid numerical instability.")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()

# Calendar date selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2008, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Price type selection
price_type = st.sidebar.selectbox("Select Price Type", ["High", "Low", "Open", "Close", "Adj Close"])
degree = st.sidebar.slider("Polynomial Degree", 1, 10, 3)

# ARIMA parameters
st.sidebar.header("ARIMA Parameters")
p_range = st.sidebar.slider("P (AR) Range", 0, 5, (0, 2))
q_range = st.sidebar.slider("Q (MA) Range", 0, 5, (0, 2))
d_range = st.sidebar.slider("D (Differencing) Range", 0, 2, (0, 1))
forecast_days = st.sidebar.slider("Forecast Days", 1, 365, 30)

run_analysis_btn = st.sidebar.button("Run Complete Analysis")

# Function to detect currency based on ticker
def detect_currency(ticker):
    # Indian stock indicators
    indian_indicators = ['.NS', '.BO', '.NSE', '.BSE', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'HDFCBANK', 
                         'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'ITC', 'LT', 'BHARTIARTL']
    
    # Check if ticker contains Indian indicators
    if any(indicator in ticker.upper() for indicator in indian_indicators):
        return "₹"
    else:
        return "$"

def safe_ljungbox(resids, max_lag=10):
    n = len(resids)
    # choose a safe lag: at most n-1 and at most requested max_lag
    lag = min(max_lag, max(1, n - 1))
    try:
        result = acorr_ljungbox(resids, lags=[lag], return_df=True)
        pval = float(result["lb_pvalue"].iloc[0])
        stat = float(result["lb_stat"].iloc[0]) if "lb_stat" in result.columns else float(result["lb_value"].iloc[0])
        return stat, pval, None
    except Exception as ex:
        return None, None, str(ex)

def safe_jarque_bera(resids):
    try:
        jb_stat, jb_p = jarque_bera(resids)
        return float(jb_stat), float(jb_p), None
    except Exception as ex:
        return None, None, str(ex)

def safe_kpss(data):
    try:
        # KPSS test with different regression types
        kpss_stat, p_value, lags, critical_values = kpss(data, regression='ct', nlags='auto')
        
        # Manual p-value calculation based on critical values
        cv_1pct = critical_values['1%']
        cv_5pct = critical_values['5%'] 
        cv_10pct = critical_values['10%']
        
        # Determine p-value based on test statistic and critical values
        if kpss_stat > cv_1pct:
            manual_pvalue = 0.01
        elif kpss_stat > cv_5pct:
            manual_pvalue = 0.05
        elif kpss_stat > cv_10pct:
            manual_pvalue = 0.10
        else:
            manual_pvalue = 0.50
            
        return float(kpss_stat), float(manual_pvalue), critical_values, None
    except Exception as ex:
        return None, None, None, str(ex)

def safe_adfuller(data):
    try:
        adf_result = adfuller(data)
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
        return adf_stat, adf_p, None
    except Exception as ex:
        return None, None, str(ex)

def fit_arima_model(data, p, d, q):
    try:
        model = SARIMAX(data, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
        fitted_model = model.fit(disp=False)
        return fitted_model, None
    except Exception as ex:
        return None, str(ex)

if run_analysis_btn:
    st.header(f"Complete Analysis for {ticker}")
    
    # Auto-detect currency
    currency_symbol = detect_currency(ticker)
    st.sidebar.info(f"Detected Currency: {currency_symbol}")

    # download data
    try:
        with st.spinner(f"Downloading {ticker}..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data is None or data.empty or price_type not in data.columns:
            st.error(f"No data or '{price_type}' column not found. Check ticker or date range.")
            st.stop()

        price_data = data[price_type].copy()
        price_data = price_data.dropna()
        n = len(price_data)
        if n < 10:
            st.error(f"Not enough data points ({n}). Increase date range or pick another ticker.")
            st.stop()

        # KPSS Test on original data
        st.subheader("KPSS Test - Stationarity Check (Original Data)")
        kpss_stat, kpss_p, kpss_critical_values, kpss_err = safe_kpss(price_data)
        
        if kpss_err:
            st.error(f"KPSS test error: {kpss_err}")
        else:
            st.write(f"**KPSS Test Statistic:** {kpss_stat:.6f}")
            st.write(f"**5% Critical Value:** {kpss_critical_values['5%']:.4f}")

            if kpss_stat > kpss_critical_values['5%']:
                st.error("✗ Data is Difference-stationary (test statistic > 5% critical value)")
            else:
                st.success("✓ Data appears Trend-stationary (test statistic < 5% critical value)")

        # show basics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Current {price_type} Price", f"{currency_symbol}{float(price_data.iloc[-1]):.2f}")
        with col2:
            st.metric("Data Points", n)
        with col3:
            actual_days = (price_data.index[-1] - price_data.index[0]).days
            st.metric("Analysis Period (days)", actual_days)

        # Prepare X: center + scale ordinal dates
        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        dates_mean = float(dates.mean(axis=0)[0])
        dates_max = float(dates.max(axis=0)[0])
        dates_min = float(dates.min(axis=0)[0])
        dates_range = dates_max - dates_min
        if dates_range == 0:
            st.error("All dates identical (unexpected).")
            st.stop()
        X = (dates - dates_mean) / dates_range

        y = price_data.values.astype(float)

        # Build polynomial features and fit model
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        st.subheader("Polynomial Model Performance")
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{currency_symbol}{rmse:.4f}")
        c2.metric("R²", f"{r2:.4f}")

        # Plot actual vs predicted
        st.subheader(f"Actual vs Predicted (Degree = {degree})")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(price_data.index, y, label=f"Actual {price_type}", linewidth=2)
        ax.plot(price_data.index, y_pred, label="Predicted", linestyle="--", linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Price ({currency_symbol})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Residuals
        residuals = y - y_pred
        st.subheader("Residual Analysis")

        # Residual time plot
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(price_data.index, residuals, label="Residuals")
        ax.axhline(0, linestyle="--", color="k")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Residual ({currency_symbol})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # ACF and PACF plots for residuals
        st.subheader("ACF and PACF Plots for Residuals")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(residuals, ax=ax1, lags=20)
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(residuals, ax=ax2, lags=20)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        plt.tight_layout()
        st.pyplot(fig)

        # ARIMA Analysis on Residuals
        st.header("ARIMA Analysis on Residuals")
        
        st.write(f"**ARIMA Parameters Range:**")
        st.write(f"- P (AR): {p_range[0]} to {p_range[1]}")
        st.write(f"- D (Differencing): {d_range[0]} to {d_range[1]}")
        st.write(f"- Q (MA): {q_range[0]} to {q_range[1]}")
        
        results = []
        with st.spinner("Fitting ARIMA models on residuals..."):
            for p in range(p_range[0], p_range[1] + 1):
                for d in range(d_range[0], d_range[1] + 1):
                    for q in range(q_range[0], q_range[1] + 1):
                        try:
                            model_arima, error = fit_arima_model(residuals, p, d, q)
                            if model_arima is not None:
                                aic = model_arima.aic
                                bic = model_arima.bic
                                results.append({
                                    'p': p,
                                    'd': d, 
                                    'q': q,
                                    'AIC': aic,
                                    'BIC': bic
                                })
                        except:
                            continue
        
        if results:
            # Convert to DataFrame and sort by AIC
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('AIC')
            
            st.subheader("ARIMA Model Comparison (Sorted by AIC)")
            st.dataframe(results_df)
            
            # Best model
            best_model = results_df.iloc[0]
            st.subheader("Best ARIMA Model for Residuals")
            st.write(f"**ARIMA({best_model['p']},{best_model['d']},{best_model['q']})**")
            st.write(f"**AIC:** {best_model['AIC']:.2f}")
            st.write(f"**BIC:** {best_model['BIC']:.2f}")
            
            # Fit the best ARIMA model
            best_arima_model, error = fit_arima_model(residuals, int(best_model['p']), int(best_model['d']), int(best_model['q']))
            
            if best_arima_model:
                # Combined Forecast: Trend + ARIMA on residuals
                st.header("Combined Forecast: Trend + ARIMA")
                
                # Generate future dates for trend forecast
                last_date = price_data.index[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                future_dates_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1).astype(float)
                
                # Trend forecast (polynomial)
                future_X = (future_dates_ord - dates_mean) / dates_range
                future_X_poly = poly.transform(future_X)
                trend_forecast = model.predict(future_X_poly)
                
                # ARIMA forecast on residuals
                arima_forecast = best_arima_model.get_forecast(steps=forecast_days)
                residual_forecast = arima_forecast.predicted_mean
                residual_ci = arima_forecast.conf_int()
                
                # Combined forecast
                combined_forecast = trend_forecast + residual_forecast
                combined_ci_lower = trend_forecast + residual_ci.iloc[:, 0]
                combined_ci_upper = trend_forecast + residual_ci.iloc[:, 1]
                
                # Plot combined forecast
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Historical data
                ax.plot(price_data.index, y, label='Historical Data', linewidth=2)
                ax.plot(price_data.index, y_pred, label='Polynomial Fit', linestyle='--', linewidth=1.5)
                
                # Forecasts
                ax.plot(future_dates, combined_forecast, label='Combined Forecast', color='red', linewidth=2)
                ax.plot(future_dates, trend_forecast, label='Trend Forecast', color='green', linestyle='--', linewidth=1.5)
                ax.fill_between(future_dates, combined_ci_lower, combined_ci_upper, color='pink', alpha=0.3, label='Confidence Interval')
                
                ax.set_xlabel('Date')
                ax.set_ylabel(f'Price ({currency_symbol})')
                ax.set_title(f'Combined Forecast: Polynomial Trend + ARIMA on Residuals')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Forecast details
                st.subheader("Forecast Details")
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Trend_Forecast': trend_forecast,
                    'Residual_Forecast': residual_forecast,
                    'Combined_Forecast': combined_forecast,
                    'CI_Lower': combined_ci_lower,
                    'CI_Upper': combined_ci_upper
                })
                forecast_df[f'Trend_Forecast ({currency_symbol})'] = forecast_df['Trend_Forecast'].round(2)
                forecast_df[f'Residual_Forecast ({currency_symbol})'] = forecast_df['Residual_Forecast'].round(2)
                forecast_df[f'Combined_Forecast ({currency_symbol})'] = forecast_df['Combined_Forecast'].round(2)
                forecast_df[f'CI_Lower ({currency_symbol})'] = forecast_df['CI_Lower'].round(2)
                forecast_df[f'CI_Upper ({currency_symbol})'] = forecast_df['CI_Upper'].round(2)
                
                st.dataframe(forecast_df[['Date', f'Trend_Forecast ({currency_symbol})', 
                                        f'Residual_Forecast ({currency_symbol})', 
                                        f'Combined_Forecast ({currency_symbol})',
                                        f'CI_Lower ({currency_symbol})', 
                                        f'CI_Upper ({currency_symbol})']])
                
                # Next day forecast
                st.subheader("Next Day Forecast")
                next_day_trend = trend_forecast[0]
                next_day_residual = residual_forecast[0]
                next_day_combined = combined_forecast[0]
                current_price = float(y[-1])
                change = next_day_combined - current_price
                change_pct = (change / current_price) * 100.0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trend Component", f"{currency_symbol}{next_day_trend:.2f}")
                with col2:
                    st.metric("Residual Component", f"{currency_symbol}{next_day_residual:.2f}")
                with col3:
                    st.metric("Combined Forecast", f"{currency_symbol}{next_day_combined:.2f}", 
                             delta=f"{change:.2f} ({change_pct:.2f}%)")
                
        else:
            st.error("No ARIMA models could be fitted with the selected parameters.")

    except Exception as main_ex:
        st.error(f"Main pipeline error: {main_ex}")
        st.info("Try a smaller degree, shorter date range, or different ticker.")
