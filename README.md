<div align="center">

# 💸 AI Expense Tracker — Backend

### *Your intelligent financial companion powered by Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)
[![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red?style=for-the-badge)](https://github.com/Yashvardhansingh2)

<br/>

> A blazing-fast Flask REST API that uses AI & Machine Learning to automatically categorize expenses, detect spending anomalies, summarize your finances, and forecast future budgets — all in real time.

</div>

---

## ✨ Features

- 🤖 **AI-Powered Categorization** — Automatically classifies expenses into 10 smart categories using a trained Naive Bayes ML model
- 🔑 **Keyword Intelligence** — Lightning-fast keyword-based fallback for common merchants (Zomato, Uber, Netflix, and more)
- 📊 **Expense Summarization** — Generates a percentage-based breakdown of your spending across all categories
- 🚨 **Anomaly Detection** — Identifies unusual or suspicious transactions using statistical analysis (2σ threshold)
- 🔮 **Budget Forecasting** — Predicts next month's spending using historical expense data
- 🌐 **CORS-Enabled** — Ready to plug into any frontend (React, Vue, Flutter, etc.)
- ⚡ **Lightweight & Fast** — Minimal dependencies, production-ready with Gunicorn

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🐍 **Language** | Python 3.8+ |
| 🌶️ **Framework** | Flask |
| 🤖 **ML Library** | scikit-learn (Naive Bayes, CountVectorizer) |
| 📦 **Model Persistence** | joblib |
| 📐 **Data Processing** | pandas, numpy |
| 🔗 **CORS** | flask-cors |
| 🚀 **Production Server** | Gunicorn |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- **Python 3.8+** → [Download](https://www.python.org/downloads/)
- **pip** (comes bundled with Python)
- **Git** → [Download](https://git-scm.com/)

---

### 📥 Installation

**1. Clone the repository**

```bash
git clone https://github.com/Yashvardhansingh2/ai_expense_tracker_backend.git
cd ai_expense_tracker_backend
```

**2. Create and activate a virtual environment**

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Train the ML model**

```bash
python train_model.py
```

> This generates `model.pkl` and `vectorizer.pkl` used by the API.

**5. Run the development server**

```bash
python app.py
```

The server starts at **`http://localhost:5001`** 🎉

**6. (Optional) Run with Gunicorn for production**

```bash
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

---

### 🔐 Environment Setup

Create a `.env` file in the root directory:

```env
# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Server settings
HOST=0.0.0.0
PORT=5001

# (Optional) Secret key for session management
SECRET_KEY=your-super-secret-key-here
```

---

## 📡 API Endpoints

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|-------------|
| `POST` | `/predict` | 🤖 Predict expense category | `{ "text": "Zomato order" }` |
| `POST` | `/summarize` | 📊 Summarize expenses by category | `{ "expenses": [{ "category": "Food", "amount": 500 }] }` |
| `POST` | `/detect_anomalies` | 🚨 Detect unusual transactions | `{ "expenses": [{ "amount": 100 }, ...] }` |
| `POST` | `/forecast` | 🔮 Forecast next month's budget | `{ "expenses": [{ "amount": 200 }, ...] }` |

---

### 📬 Example Requests & Responses

<details>
<summary><b>POST /predict</b> — Categorize an expense</summary>

**Request:**
```json
{
  "text": "Uber ride to airport"
}
```

**Response:**
```json
{
  "category": "Transportation"
}
```
</details>

<details>
<summary><b>POST /summarize</b> — Get spending summary</summary>

**Request:**
```json
{
  "expenses": [
    { "category": "Food & Dining", "amount": 1500 },
    { "category": "Transportation", "amount": 500 },
    { "category": "Entertainment", "amount": 300 }
  ]
}
```

**Response:**
```json
{
  "summary": "Expense Summary — Food & Dining: 65.22% | Transportation: 21.74% | Entertainment: 13.04%. Total spent: ₹2300",
  "percentages": {
    "Food & Dining": 65.22,
    "Transportation": 21.74,
    "Entertainment": 13.04
  }
}
```
</details>

<details>
<summary><b>POST /detect_anomalies</b> — Find unusual spending</summary>

**Request:**
```json
{
  "expenses": [
    { "description": "Lunch", "amount": 200 },
    { "description": "Grocery", "amount": 300 },
    { "description": "Laptop", "amount": 85000 }
  ]
}
```

**Response:**
```json
{
  "anomalies": [{ "description": "Laptop", "amount": 85000 }],
  "count": 1
}
```
</details>

<details>
<summary><b>POST /forecast</b> — Predict next month's budget</summary>

**Request:**
```json
{
  "expenses": [
    { "amount": 500 },
    { "amount": 1200 },
    { "amount": 800 }
  ]
}
```

**Response:**
```json
{
  "forecast": 2750.0
}
```
</details>

---

## 📁 Project Structure

```
ai_expense_tracker_backend/
│
├── 📄 app.py               # Main Flask application & all API routes
├── 🧠 train_model.py       # ML model training script (Naive Bayes)
├── 📦 model.pkl            # Trained classification model (auto-generated)
├── 📦 vectorizer.pkl       # CountVectorizer for text features (auto-generated)
├── 📋 requirements.txt     # Python dependencies
├── 📖 README.md            # Project documentation
└── 🐍 venv/                # Virtual environment (not committed)
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**!

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

```
MIT License — feel free to use, modify, and distribute this project.
```

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Made with ❤️ by [Yashvardhan Singh](https://github.com/Yashvardhansingh2)**

<br/>

*If you have any questions, feel free to open an issue or reach out!*

</div>
