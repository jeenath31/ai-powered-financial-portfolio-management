# predict_lstm.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load your pre-trained LSTM model
model = load_model("lstm_model.h5")
scaler = MinMaxScaler()

def predict_future_prices(prices, steps=5):
    data = np.array(prices).reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)

    look_back = 10
    if len(scaled_data) < look_back:
        raise ValueError("Not enough data for prediction window.")

    inputs = scaled_data[-look_back:].reshape(1, look_back, 1)

    predictions = []
    for _ in range(steps):
        pred = model.predict(inputs, verbose=0)
        predictions.append(pred[0][0])
        pred_reshaped = np.reshape(pred[0][0], (1, 1, 1))  # FIXED: correct shape (1, 1, 1)
        inputs = np.concatenate((inputs[:, 1:, :], pred_reshaped), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
    return predicted_prices
