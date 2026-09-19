import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

training_data = [
    # Food & Dining
    ('zomato order', 'Food & Dining'),
    ('swiggy dinner', 'Food & Dining'),
    ('restaurant lunch', 'Food & Dining'),
    ('dominos pizza', 'Food & Dining'),
    ('mcdonalds burger', 'Food & Dining'),
    ('starbucks coffee', 'Food & Dining'),
    ('chai and snacks', 'Food & Dining'),
    ('groceries at supermarket', 'Food & Dining'),
    ('vegetables and fruits', 'Food & Dining'),
    ('bakery bread and milk', 'Food & Dining'),
    ('dinner with friends', 'Food & Dining'),
    ('breakfast buffet', 'Food & Dining'),
    ('subway sandwich', 'Food & Dining'),
    ('kfc chicken', 'Food & Dining'),

    # Transportation
    ('uber ride to office', 'Transportation'),
    ('ola cab', 'Transportation'),
    ('metro card recharge', 'Transportation'),
    ('bus ticket', 'Transportation'),
    ('petrol fuel refill', 'Transportation'),
    ('diesel fuel', 'Transportation'),
    ('toll plaza payment', 'Transportation'),
    ('car parking fee', 'Transportation'),
    ('auto rickshaw fare', 'Transportation'),
    ('bike service and oil', 'Transportation'),

    # Housing & Utilities
    ('monthly house rent', 'Housing & Utilities'),
    ('electricity power bill', 'Housing & Utilities'),
    ('water bill payment', 'Housing & Utilities'),
    ('piped gas cylinder bill', 'Housing & Utilities'),
    ('wifi broadband internet bill', 'Housing & Utilities'),
    ('phone mobile recharge', 'Housing & Utilities'),
    ('society maintenance fee', 'Housing & Utilities'),
    ('house cleaning maid salary', 'Housing & Utilities'),
    ('plumber repair', 'Housing & Utilities'),

    # Entertainment
    ('netflix monthly subscription', 'Entertainment'),
    ('movie cinema tickets', 'Entertainment'),
    ('spotify music premium', 'Entertainment'),
    ('amazon prime subscription', 'Entertainment'),
    ('disney hotstar subscription', 'Entertainment'),
    ('playstation video games', 'Entertainment'),
    ('steam game purchase', 'Entertainment'),
    ('concert music tickets', 'Entertainment'),
    ('amusement park tickets', 'Entertainment'),

    # Health & Fitness
    ('gym monthly membership', 'Health & Fitness'),
    ('doctor clinic consultation', 'Health & Fitness'),
    ('pharmacy medicine tablets', 'Health & Fitness'),
    ('dental clinic checkup', 'Health & Fitness'),
    ('blood test laboratory', 'Health & Fitness'),
    ('hospital treatment bill', 'Health & Fitness'),
    ('yoga class subscription', 'Health & Fitness'),
    ('protein powder supplement', 'Health & Fitness'),

    # Shopping
    ('amazon online shopping', 'Shopping'),
    ('flipkart order', 'Shopping'),
    ('myntra clothes purchase', 'Shopping'),
    ('zara shirt and trousers', 'Shopping'),
    ('nike running shoes', 'Shopping'),
    ('electronics gadgets store', 'Shopping'),
    ('headphones earphone purchase', 'Shopping'),
    ('watch and sunglasses', 'Shopping'),
    ('home decor furniture', 'Shopping'),

    # Education
    ('college tuition semester fee', 'Education'),
    ('school quarterly fee', 'Education'),
    ('udemy online course', 'Education'),
    ('coursera certificate', 'Education'),
    ('textbooks and notebooks', 'Education'),
    ('coding bootcamp subscription', 'Education'),
    ('library membership', 'Education'),

    # Travel
    ('flight airline tickets', 'Travel'),
    ('hotel resort stay booking', 'Travel'),
    ('airbnb vacation booking', 'Travel'),
    ('train irctc ticket booking', 'Travel'),
    ('vacation tour package', 'Travel'),
    ('visa application fee', 'Travel'),
    ('luggage bag purchase', 'Travel'),

    # Financial
    ('home loan emi payment', 'Financial'),
    ('car loan installment', 'Financial'),
    ('credit card bill payment', 'Financial'),
    ('health insurance premium', 'Financial'),
    ('life insurance term plan', 'Financial'),
    ('mutual fund sip investment', 'Financial'),
    ('stock share purchase', 'Financial'),
    ('bank account service charges', 'Financial'),
    ('income tax advance payment', 'Financial'),
]

df = pd.DataFrame(training_data, columns=['description', 'category'])

vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=True)
X = vectorizer.fit_transform(df['description'])
y = df['category']

model = MultinomialNB(alpha=0.1)
model.fit(X, y)

model_path = os.path.join(BASE_DIR, 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model trained with {len(df)} samples across {df['category'].nunique()} categories.")
print(f"Saved to {model_path} and {vectorizer_path}")
