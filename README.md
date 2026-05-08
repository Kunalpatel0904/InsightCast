# InsightCast — Smart Business Analytics Dashboard
## Kunal Patel (70012200155) | NMIMS MPSTME Shirpur | 2025-26

---

## ⚡ Quick Start (follow these steps exactly)

### Step 1 — Install Anaconda
Download from: https://www.anaconda.com/download
Choose Python 3.10 installer for Windows.

### Step 2 — Open Anaconda Prompt
Search "Anaconda Prompt" in the Windows Start menu.

### Step 3 — Create & activate virtual environment
```
conda create -n insightcast python=3.10
conda activate insightcast
```

### Step 4 — Navigate to project folder
```
cd Desktop\insightcast
```

### Step 5 — Install all dependencies
```
pip install -r requirements.txt
```

### Step 6 — Generate sample dataset (run once)
```
python generate_data.py
```

### Step 7 — Launch dashboard
```
streamlit run app.py
```

The dashboard opens automatically at:  http://localhost:8501

---

## 📁 Project Structure
```
insightcast/
├── app.py                ← Main Streamlit dashboard
├── preprocessing.py      ← Data cleaning + feature engineering
├── churn_model.py        ← XGBoost classifier + SHAP
├── clustering.py         ← K-Means segmentation
├── forecasting.py        ← Revenue time-series forecasting
├── generate_data.py      ← Generates sample dataset
├── requirements.txt      ← All Python dependencies
└── data/
    └── ecommerce.csv     ← Auto-generated dataset
```

---

## 🎯 Dashboard Tabs
| Tab | What it shows |
|-----|---------------|
| 📊 KPI Overview | Total customers, churn rate, revenue, trends |
| 🚨 Churn Predictor | At-risk customers table, scatter map |
| 📈 Revenue Forecast | 6-month forecast with confidence bands |
| 👥 Customer Segments | K-Means clusters, radar chart |
| 🔍 Model Explainability | SHAP beeswarm, waterfall, confusion matrix |

---

## 🛠️ Tech Stack
- Python 3.10, Streamlit, XGBoost, SHAP, Scikit-learn
- Pandas, NumPy, Plotly, Matplotlib
