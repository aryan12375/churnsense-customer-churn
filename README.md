# ChurnSense 🎯

> **End-to-end customer churn prediction system** — from raw data to explainable, production-ready predictions with a live stakeholder dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## Overview

ChurnSense predicts which telecom customers are most likely to cancel their subscriptions — and **explains exactly why**, so retention teams can act before it's too late.

The system goes beyond accuracy metrics. It integrates **SHAP-based explainability** directly into an interactive business dashboard, bridging the gap between ML research and real-world deployment.

### What makes this different?

| Feature | Typical Churn Project | ChurnSense |
|---|---|---|
| Explainability | SHAP plots as afterthought | SHAP force plots per customer, built into the UI |
| Model coverage | 1–2 models | 5 models including ANN + full comparison |
| Production readiness | Notebook only | Modular src/, MLOps retraining pipeline, versioned artifacts |
| Dashboard | None | React dashboard with risk tiers, filters, CSV export |
| Reproducibility | None | MLflow tracking, seed-locked runs |

---

## Project Structure

```
ChurnSense/
├── data/
│   └── telco_churn.csv          # Kaggle Telco Customer Churn dataset
├── src/
│   ├── preprocessing.py         # Full data cleaning, encoding, SMOTE, engineering
│   ├── train.py                 # Model training runner (all 5 models)
│   ├── evaluate.py              # Metrics, confusion matrix, ROC curves
│   ├── shap_explain.py          # Global + local SHAP analysis
│   ├── ann_model.py             # PyTorch ANN with early stopping
│   └── mlops_pipeline.py        # Retraining scheduler, drift detection, versioning
├── models/                      # Saved .pkl / .pt artifacts (git-ignored)
├── reports/                     # Generated charts and HTML reports
├── notebooks/
│   └── exploratory_analysis.ipynb
├── dashboard/                   # React stakeholder dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── RiskTable.jsx
│   │   │   ├── ShapChart.jsx
│   │   │   ├── MetricCards.jsx
│   │   │   ├── RiskDistribution.jsx
│   │   │   └── CustomerModal.jsx
│   │   ├── hooks/
│   │   │   └── useChurnData.js
│   │   ├── utils/
│   │   │   └── formatters.js
│   │   └── App.jsx
│   ├── package.json
│   └── index.html
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ChurnSense.git
cd ChurnSense
pip install -r requirements.txt
```

### 2. Add the dataset

Download the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle and place it at:

```
data/telco_churn.csv
```

### 3. Run the full pipeline

```bash
# Step 1 — Preprocess, train all models, evaluate, explain
python src/train.py

# Step 2 — Generate SHAP visualisations
python src/shap_explain.py

# Step 3 — (Optional) Trigger manual retrain with drift check
python src/mlops_pipeline.py --check-drift
```

### 4. Launch the dashboard

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:5173
```

---

## Model Performance

| Model | ROC-AUC | F1-Score | Training Time |
|---|---|---|---|
| Logistic Regression | 0.81 | 0.59 | ~1s |
| SVM | 0.79 | 0.57 | ~8s |
| Random Forest | 0.84 | 0.61 | ~12s |
| Gradient Boosting  | **0.86** | **0.63** | ~18s |
| ANN (PyTorch) | 0.85 | 0.62 | ~45s |

> **Champion model**: Gradient Boosting — best ROC-AUC and F1, strong on imbalanced data, fastest to explain with SHAP.

---

## Key Findings (SHAP Analysis)

The top predictors driving churn, in order of impact:

1. **Contract type (Month-to-month)** — highest single risk factor
2. **Tenure (0–12 months)** — new customers churn at 3× the rate of long-term ones
3. **Internet service (Fiber optic)** — correlated with higher charges and churn
4. **Total charges** — higher cumulative spend reduces churn risk
5. **Tech support (No)** — customers without support are significantly more likely to leave

---

## MLOps Retraining Pipeline

`src/mlops_pipeline.py` implements:

- **Population Stability Index (PSI)** — detects distribution drift in incoming data
- **Automatic retrain trigger** — kicks off when PSI > 0.2 on any key feature
- **Model versioning** — saves timestamped `.pkl` artifacts with metadata JSON
- **Performance comparison** — new model must beat champion on F1 before promotion

---

## Architecture Decisions

- **SMOTE over class_weight** — better recall on the minority churn class in benchmarks
- **Gradient Boosting as champion** — outperforms Random Forest on this dataset; interpretable with SHAP unlike SVM or ANN
- **ANN as challenger** — PyTorch MLP captures non-linear interactions; tracked alongside GB for ongoing comparison
- **React dashboard over Streamlit** — more appropriate for a real business stakeholder tool; supports proper state management and deployment

---

## Authors

- **Aryan Nair** — Data preprocessing, feature engineering, LR/SVM, model evaluation, ANN
- **Aviral** — Literature review, SMOTE, Random Forest/GBM, SHAP analysis, dashboard design

Guided by **Dr. Narendra V G**, MIT Manipal

---

## License

MIT — free to use, adapt, and build on.
