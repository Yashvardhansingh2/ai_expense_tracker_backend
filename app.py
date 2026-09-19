import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load ML model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Successfully loaded model.pkl and vectorizer.pkl")
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    model = None
    vectorizer = None

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'ai-expense-tracker-backend',
        'model_loaded': model is not None and vectorizer is not None
    })

# Known categories
known_categories = [
    "Food & Dining", "Transportation", "Housing & Utilities", "Entertainment",
    "Health & Fitness", "Shopping", "Education", "Travel", "Financial", "Others"
]

# Keyword mapping
keyword_map = {
    # Food & Dining
    "zomato": "Food & Dining",
    "swiggy": "Food & Dining",
    "restaurant": "Food & Dining",
    "pizza": "Food & Dining",
    "burger": "Food & Dining",
    "coffee": "Food & Dining",
    "tea": "Food & Dining",
    "chai": "Food & Dining",
    "snack": "Food & Dining",
    "groceries": "Food & Dining",
    "grocery": "Food & Dining",
    "supermarket": "Food & Dining",
    "dinner": "Food & Dining",
    "lunch": "Food & Dining",
    "breakfast": "Food & Dining",
    "cafe": "Food & Dining",
    "food": "Food & Dining",
    "dining": "Food & Dining",

    # Transportation
    "fuel": "Transportation",
    "petrol": "Transportation",
    "diesel": "Transportation",
    "uber": "Transportation",
    "ola": "Transportation",
    "cab": "Transportation",
    "taxi": "Transportation",
    "bus": "Transportation",
    "train": "Transportation",
    "metro": "Transportation",
    "auto": "Transportation",
    "toll": "Transportation",
    "parking": "Transportation",

    # Housing & Utilities
    "rent": "Housing & Utilities",
    "electricity": "Housing & Utilities",
    "water": "Housing & Utilities",
    "gas": "Housing & Utilities",
    "wifi": "Housing & Utilities",
    "internet": "Housing & Utilities",
    "broadband": "Housing & Utilities",
    "maintenance": "Housing & Utilities",
    "bill": "Housing & Utilities",

    # Entertainment
    "movie": "Entertainment",
    "cinema": "Entertainment",
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "prime": "Entertainment",
    "hotstar": "Entertainment",
    "game": "Entertainment",
    "gaming": "Entertainment",
    "concert": "Entertainment",

    # Health & Fitness
    "gym": "Health & Fitness",
    "doctor": "Health & Fitness",
    "medicine": "Health & Fitness",
    "pharmacy": "Health & Fitness",
    "hospital": "Health & Fitness",
    "clinic": "Health & Fitness",
    "dental": "Health & Fitness",
    "dentist": "Health & Fitness",

    # Shopping
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "myntra": "Shopping",
    "clothes": "Shopping",
    "clothing": "Shopping",
    "shoes": "Shopping",
    "mall": "Shopping",

    # Education
    "book": "Education",
    "course": "Education",
    "udemy": "Education",
    "coursera": "Education",
    "tuition": "Education",
    "school": "Education",
    "college": "Education",

    # Travel
    "flight": "Travel",
    "airline": "Travel",
    "hotel": "Travel",
    "resort": "Travel",
    "airbnb": "Travel",
    "vacation": "Travel",
    "trip": "Travel",

    # Financial
    "loan": "Financial",
    "emi": "Financial",
    "credit": "Financial",
    "insurance": "Financial",
    "investment": "Financial",
    "sip": "Financial",
    "mutual fund": "Financial",
    "tax": "Financial"
}

category_aliases = {
    "food": "Food & Dining",
    "dining": "Food & Dining",
    "utilities": "Housing & Utilities",
    "housing": "Housing & Utilities",
    "transport": "Transportation",
    "fitness": "Health & Fitness",
    "health": "Health & Fitness",
    "finance": "Financial",
}

def normalize_category(cat: str) -> str:
    if not cat:
        return "Others"
    cat_lower = cat.lower().strip()
    if cat in known_categories:
        return cat
    if cat_lower in category_aliases:
        return category_aliases[cat_lower]
    for known in known_categories:
        if cat_lower in known.lower():
            return known
    return cat.title()

@app.route('/predict', methods=['POST'])
def predict_category():
    data = request.get_json() or {}
    text = data.get('text', '').lower().strip()

    if not text:
        return jsonify({'category': 'Others'})

    # Step 1: Keyword match (match whole words or substrings)
    for keyword, category in keyword_map.items():
        if keyword in text:
            return jsonify({'category': category})

    # Step 2: ML Model prediction
    if model and vectorizer:
        try:
            vector = vectorizer.transform([text])
            prediction = model.predict(vector)[0]
            norm = normalize_category(prediction)
            if norm in known_categories:
                return jsonify({'category': norm})
        except Exception as e:
            print("Model error:", e)

    # Step 3: Dynamic fallback category
    return jsonify({'category': text.title() if text else "Others"})


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
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
