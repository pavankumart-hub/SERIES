# simple_test.py
import streamlit as st

st.title("Simple Test App")
st.write("If you see this, Streamlit is working!")

try:
    import yfinance as yf
    st.write("✅ yfinance imported successfully")
except Exception as e:
    st.write(f"❌ yfinance import failed: {e}")

try:
    import pandas as pd
    st.write("✅ pandas imported successfully")
except Exception as e:
    st.write(f"❌ pandas import failed: {e}")
