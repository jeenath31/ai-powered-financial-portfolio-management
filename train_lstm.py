import yfinance as yf 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# List of stock tickers (20 companies)
tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
    "ICICIBANK.NS", "LT.NS", "SBIN.NS", "BAJAJ-AUTO.NS", 
    "MARUTI.NS", "HEROMOTOCO.NS", "BHARTIARTL.NS", "TATASTEEL.NS", 
    "ADANIENT.NS", "ITC.NS", "JSWSTEEL.NS", "POWERGRID.NS", 
    "COALINDIA.NS", "ONGC.NS", "WIPRO.NS", "TECHM.NS"
]

def get_data(ticker):
    data = yf.Ticker(ticker).history(period="6mo")
    return data["Close"].values.reshape(-1, 1)

def create_dataset(data, step=60):
    X, y = [], []
    for i in range(len(data)-step-1):
        X.append(data[i:i+step, 0])
        y.append(data[i+step, 0])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
X_all, y_all = [], []

for ticker in tickers:
    close_prices = get_data(ticker)
    scaled = scaler.fit_transform(close_prices)
    X, y = create_dataset(scaled)
    X_all.append(X)
    y_all.append(y)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

# Define and train the model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_all.shape[1], 1)),
    Dropout(0.3),
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(50),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Early Stopping callback
early_stop = EarlyStopping(monitor='loss', patience=10)

# Train the model
model.fit(X_all, y_all, epochs=50, batch_size=16, callbacks=[early_stop])

# Save the model
model.save("lstm_model.h5")
