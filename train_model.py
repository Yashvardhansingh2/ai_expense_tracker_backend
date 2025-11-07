import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Step 1: Sample training data
data = {
    'description': [
        'Zomato order', 'Swiggy dinner', 'Uber ride', 'Ola trip',
        'Electricity bill', 'Netflix subscription', 'Phone recharge', 'Flight ticket',
        'Groceries purchase', 'Amazon shopping', 'Water bill', 'Restaurant lunch'
    ],
    'category': [
        'Food', 'Food', 'Travel', 'Travel',
        'Utilities', 'Entertainment', 'Utilities', 'Travel',
        'Food', 'Shopping', 'Utilities', 'Food'
    ]
}

df = pd.DataFrame(data)

# Step 2: Train AI model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['description'])
y = df['category']

model = MultinomialNB()
model.fit(X, y)

# Step 3: Save model and vectorizer for Flask
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model trained and saved successfully!")
