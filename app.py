import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json

app = Flask(__name__)
CORS(app)

# Load your ML model and vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    model = None
    vectorizer = None

# Known categories
known_categories = [
    "Food & Dining", "Transportation", "Housing & Utilities", "Entertainment",
    "Health & Fitness", "Shopping", "Education", "Travel", "Financial", "Others"
]

# Keyword mapping
keyword_map = {
    "rent": "Housing & Utilities",
    "electricity": "Housing & Utilities",
    "wifi": "Housing & Utilities",
    "internet": "Housing & Utilities",
    "fuel": "Transportation",
    "uber": "Transportation",
    "bus": "Transportation",
    "train": "Transportation",
    "movie": "Entertainment",
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "gym": "Health & Fitness",
    "doctor": "Health & Fitness",
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "zomato": "Food & Dining",
    "swiggy": "Food & Dining",
    "restaurant": "Food & Dining",
    "book": "Education",
    "course": "Education",
    "loan": "Financial",
    "emi": "Financial",
    "credit": "Financial",
    "insurance": "Financial"
}


@app.route('/predict', methods=['POST'])
def predict_category():
    data = request.get_json()
    text = data.get('text', '').lower().strip()

    # Step 1: Keyword match
    for keyword, category in keyword_map.items():
        if keyword in text:
            return jsonify({'category': category})

    # Step 2: ML Model prediction
    if model and vectorizer:
        try:
            vector = vectorizer.transform([text])
            prediction = model.predict(vector)[0]
            if prediction in known_categories:
                return jsonify({'category': prediction})
        except Exception as e:
            print("Model error:", e)

    # Step 3: Dynamic new category
    category = text.title() if text else "Others"
    return jsonify({'category': category})


@app.route('/summarize', methods=['POST'])
def summarize_expenses():
    data = request.get_json()
    df = pd.DataFrame(data['expenses'])  # [{'category':..., 'amount':...}, ...]

    summary = df.groupby('category')['amount'].sum()
    total = summary.sum()

    summary_percent = {cat: round((amt / total) * 100, 2) for cat, amt in summary.items()}

    summary_text = " | ".join([f"{cat}: {pct}%" for cat, pct in summary_percent.items()])
    summary_text = f"Expense Summary — {summary_text}. Total spent: ₹{total}"

    return jsonify({'summary': summary_text, 'percentages': summary_percent})


@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    data = request.get_json()
    expenses = data.get('expenses', [])

    if len(expenses) < 3:
        return jsonify({'anomalies': [], 'message': 'Not enough data for anomaly detection.'})

    df = pd.DataFrame(expenses)

    # Simple anomaly detection: expenses that are 2 standard deviations above mean
    mean_amount = df['amount'].mean()
    std_amount = df['amount'].std()

    if std_amount == 0:
        # All amounts are the same, no anomalies
        return jsonify({'anomalies': []})

    threshold = mean_amount + 2 * std_amount
    anomalies = df[df['amount'] > threshold].to_dict(orient='records')

    return jsonify({'anomalies': anomalies, 'count': len(anomalies)})


@app.route('/forecast', methods=['POST'])
def forecast_budget():
    data = request.get_json()
    expenses = data.get('expenses', [])

    if len(expenses) < 2:
        return jsonify({'forecast': 0.0, 'message': 'Not enough data for prediction.'})

    # Group by month (assuming current month is 1, next is 2, etc.)
    # For simplicity, we'll use the current month as 1 and predict for month 2
    df = pd.DataFrame(expenses)
    total_spent = df['amount'].sum()

    # Simple forecast: average spending + 10% growth
    forecast = total_spent * 1.1

    return jsonify({'forecast': round(forecast, 2)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
