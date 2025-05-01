from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import random

# === NEW ===
from predict_lstm import predict_future_prices
import numpy as np

app = Flask(__name__)

# Load Pretrained BERT Model & Tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Load VADER Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# === NEWS ===
def fetch_news_india():
    url = "https://www.moneycontrol.com/news/business/markets/"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [item.text.strip() for item in soup.find_all("h2")[:5]]
    return headlines if headlines else ["No news available."]

def fetch_news_us():
    url = "https://finance.yahoo.com/topic/stock-market-news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [item.text.strip() for item in soup.find_all("h3")[:5]]
    return headlines if headlines else ["No news available."]

# === SENTIMENT ===
def predict_sentiment_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = torch.argmax(probs).item()
    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    return sentiment_labels[sentiment_score]

def predict_sentiment_vader(text):
    score = vader_analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def combined_sentiment_analysis(text):
    bert_sentiment = predict_sentiment_bert(text)
    vader_sentiment = predict_sentiment_vader(text)
    return "Positive" if bert_sentiment in ["Very Positive", "Positive"] else "Negative" if bert_sentiment in ["Very Negative", "Negative"] else vader_sentiment

# === STOCK DATA & LSTM PREDICTION ===
def fetch_stock_price_series(stock_symbol, market):
    try:
        if market == "india":
            if not stock_symbol.endswith(('.NS', '.BO')):
                stock_symbol += ".NS"
        stock = yf.Ticker(stock_symbol)
        close_prices = stock.history(period="6mo")["Close"].values[-90:]
        return close_prices.tolist()
    except Exception:
        return []

def assess_risk_lstm(predicted_prices):
    trend = predicted_prices[-1] - predicted_prices[0]
    if trend < -1.0:
        return "High"
    elif trend < 1.0:
        return "Medium"
    else:
        return "Low"

def calculate_confidence_lstm(risk):
    if risk == "High":
        return f"{random.randint(30, 50)}%"
    elif risk == "Medium":
        return f"{random.randint(60, 75)}%"
    else:
        return f"{random.randint(80, 95)}%"

# === ROUTES ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/add_stock", methods=["POST"])
def add_stock():
    data = request.json
    stock_symbol = data.get("stock_symbol")
    quantity = float(data.get("quantity"))
    purchase_price = float(data.get("purchase_price"))

    # Add the stock to the portfolio
    portfolio.append({
        "stock_symbol": stock_symbol,
        "quantity": quantity,
        "purchase_price": purchase_price
    })

    return jsonify({
        "message": "Stock added successfully!",
        "portfolio": portfolio
    })
@app.route("/portfolio", methods=["GET"])
def portfolio_summary():
    total_investment = 0
    total_current_value = 0
    total_pnl = 0
    detailed_holdings = []

    for stock in portfolio:  # Replace with database query in production
        stock_symbol = stock["stock_symbol"]
        quantity = stock["quantity"]
        purchase_price = stock["purchase_price"]

        # Fetch current price using yfinance
        try:
            current_price = yf.Ticker(stock_symbol).history(period="1d")["Close"].iloc[-1]
        except Exception as e:
            current_price = 0.0  # Handle cases where data fetch fails
        
        invested_amount = quantity * purchase_price
        current_value = quantity * current_price
        stock_pnl = current_value - invested_amount  # Profit/Loss calculation

        total_investment += invested_amount
        total_current_value += current_value
        total_pnl += stock_pnl
        
        detailed_holdings.append({
            "stock_symbol": stock_symbol,
            "quantity": quantity,
            "purchase_price": purchase_price,
            "current_price": current_price,
            "invested_amount": invested_amount,
            "current_value": current_value,
            "pnl": stock_pnl
        })

    return jsonify({
        "holdings": detailed_holdings,
        "total_investment": total_investment,
        "total_current_value": total_current_value,
        "overall_pnl": total_pnl
    })

portfolio = []  # Initialize as an empty list

@app.route("/get_news", methods=["POST"])
def get_news():
    market = request.form["market"]
    news = fetch_news_india() if market == "india" else fetch_news_us()
    return jsonify({"news": news})

@app.route("/predict", methods=["POST"])
\
def predict():
    news_headline = request.form["news"]
    sentiment = combined_sentiment_analysis(news_headline)
    return jsonify({"sentiment": sentiment})

@app.route("/predict_stock", methods=["POST"])
def predict_stock():
    stock_symbol = request.form["stock_symbol"]
    market = request.form["market"]

    recent_prices = fetch_stock_price_series(stock_symbol, market)
    if not recent_prices or len(recent_prices) < 60:
        return jsonify({
            "predicted_price": "Error fetching or insufficient data",
            "risk": "Unable to assess risk",
            "confidence": "N/A"
        })

    try:
        predicted_prices = predict_future_prices(recent_prices)
        risk = assess_risk_lstm(predicted_prices)
        confidence = calculate_confidence_lstm(risk)
        return jsonify({
            "predicted_price": predicted_prices[-1],
            "risk": risk,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({
            "predicted_price": "Prediction failed",
            "risk": str(e),
            "confidence": "N/A"
        })

if __name__ == "__main__":
    app.run(debug=True)
