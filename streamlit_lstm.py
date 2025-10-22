# streamlit_app_pytorch.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# Sidebar - User Inputs
# --------------------------
st.title("📈 HDFCBANK High Price Forecast using LSTM + STL (PyTorch)")

ticker = st.text_input("Enter Ticker Symbol:", "HDFCBANK.NS")
start_date = st.date_input("Start Date:", pd.to_datetime("2020-01-01"))
end_date_forecast = st.date_input("End Date:", pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
window_size = st.slider("LSTM Window Size:", min_value=5, max_value=30, value=10)
epochs = st.slider("Training Epochs:", min_value=5, max_value=50, value=10)
batch_size = 16
lr = 0.001

st.write(f"📅 Forecasting data from {start_date} to {end_date_forecast}")

# --------------------------
# Download Data
# --------------------------
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

data = load_data(ticker, start_date, end_date_forecast)
d_high = data["High"]

st.write(f"📊 Showing last 5 high prices for {ticker}")
st.dataframe(d_high.tail(5))

# --------------------------
# STL Decomposition
# --------------------------
stl = STL(d_high, period=30)
result = stl.fit()
trend = result.trend
seasonal = result.seasonal

# --------------------------
# Prepare LSTM Data
# --------------------------
def prepare_lstm_data(series, window_size):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return torch.FloatTensor(X), torch.FloatTensor(y), scaler

X_trend, y_trend, scaler_trend = prepare_lstm_data(trend, window_size)
X_seasonal, y_seasonal, scaler_seasonal = prepare_lstm_data(seasonal, window_size)

# DataLoader
train_trend = DataLoader(TensorDataset(X_trend, y_trend), batch_size=batch_size, shuffle=True)
train_seasonal = DataLoader(TensorDataset(X_seasonal, y_seasonal), batch_size=batch_size, shuffle=True)

# --------------------------
# PyTorch LSTM Model
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return out

def train_lstm(model, dataloader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for xb, yb in dataloader:
            xb = xb.unsqueeze(-1)  # Add feature dimension
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

# --------------------------
# Train Models
# --------------------------
with st.spinner("Training LSTM models..."):
    model_trend = LSTMModel()
    model_trend = train_lstm(model_trend, train_trend, epochs, lr)

    model_seasonal = LSTMModel()
    model_seasonal = train_lstm(model_seasonal, train_seasonal, epochs, lr)

# --------------------------
# Predictions
# --------------------------
def predict(model, X, scaler):
    model.eval()
    with torch.no_grad():
        X_t = X.unsqueeze(-1)
        y_pred = model(X_t).numpy()
        return scaler.inverse_transform(y_pred)

trend_pred = predict(model_trend, X_trend, scaler_trend)
seasonal_pred = predict(model_seasonal, X_seasonal, scaler_seasonal)
final_pred = trend_pred.flatten() + seasonal_pred.flatten()
actual = d_high.values[window_size:]

rmse = np.sqrt(mean_squared_error(actual, final_pred))
st.write(f"📉 RMSE (Reconstructed): {rmse:.4f}")

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(actual, label="Actual")
ax.plot(final_pred, label="Predicted (Trend + Seasonal)")
ax.set_title("LSTM Forecast on STL Components (PyTorch)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --------------------------
# Next Day Forecast
# --------------------------
def forecast_next_day(model, series, scaler):
    last_window = torch.FloatTensor(series[-window_size:]).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(last_window).numpy()
        return scaler.inverse_transform(pred_scaled)[0][0]

next_trend = forecast_next_day(model_trend, trend.values, scaler_trend)
next_seasonal = forecast_next_day(model_seasonal, seasonal.values, scaler_seasonal)

next_day_forecast_high_LSTM = next_trend + next_seasonal
st.success(f"📅 Forecast for Next Day (Trend + Seasonal): {next_day_forecast_high_LSTM:.4f}")
