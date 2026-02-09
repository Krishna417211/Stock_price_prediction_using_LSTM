import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

# 2. Cached Model Loading (Performance optimization)
@st.cache_resource
def load_my_model():
    # Ensure saved_model.h5 is in the same directory
    return load_model('saved_model.h5')

model = load_my_model()

# 3. Sidebar Inputs
st.sidebar.header("User Input Parameters")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# 4. Data Fetching
if st.sidebar.button("Predict"):
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    
    if not data.empty:
        st.subheader(f"Historical Data for {stock_ticker}")
        st.write(data.tail())

        # 5. Preprocessing (Example assuming the model uses the last 60 days)
        # Match this exactly to your training preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']])
        
        # Prepare the test sequence (last 60 days)
        if len(scaled_data) >= 60:
            last_60_days = scaled_data[-60:].reshape(1, 60, 1)
            
            # 6. Prediction
            prediction_scaled = model.predict(last_60_days)
            prediction = scaler.inverse_transform(prediction_scaled)

            # 7. Display Result
            st.metric(label=f"Predicted Next Closing Price for {stock_ticker}", 
                      value=f"${prediction[0][0]:.2f}")
            
            st.line_chart(data['Close'])
        else:
            st.error("Not enough historical data to make a prediction (need at least 60 days).")
    else:
        st.error("Could not fetch data for the given ticker.")