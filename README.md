# 💳 Credit Card Fraud Detection - Data Science Project & Interactive Dashboard

This project is a comprehensive **Data Science** study and web dashboard for **Credit Card Fraud Detection**. It is designed specifically for data science evaluation and mentor presentation, showcasing exploratory data analysis (EDA), statistical distribution analysis, class imbalance handling, machine learning benchmarking (Logistic Regression, Random Forest, XGBoost), business financial risk optimization, and live interactive fraud scoring.

---

## 🌟 Key Features

1. **Exploratory Data Analysis (EDA)**:
   - Analysis of extreme class imbalance (~0.3% fraud rate).
   - Skewness analysis & statistical summaries (Mean, Median, Std) for Transaction Amounts and Time.
   - Pearson & Spearman correlation analysis identifying top discriminative PCA features (e.g. `V14`, `V17`, `V12`, `V10`, `V4`, `V11`).
   - Boxplots and IQR outlier analysis.

2. **Machine Learning Models & Evaluation**:
   - **Logistic Regression**: Statistical baseline model with balanced class weighting.
   - **Random Forest Classifier**: Ensemble tree model capturing non-linear interactions.
   - **XGBoost Classifier**: Gradient boosted decision trees optimized with `scale_pos_weight`.
   - **Metrics Evaluated**: ROC-AUC, PR-AUC (Precision-Recall Area Under Curve), Precision, Recall, F1-Score, and Confusion Matrix (TP, FP, FN, TN).

3. **Business Financial Cost & Threshold Optimizer**:
   - Simulates real-world credit card business costs:
     - **False Negative Loss**: Undetected fraud chargeback costs (~$250 per fraud).
     - **False Positive Cost**: Manual review/audit friction costs (~$10 per alert).
   - Dynamic threshold slider (0.01 - 0.99) to find the **Optimal Decision Threshold** that minimizes total financial risk.

4. **Live Fraud Risk Inference Tool**:
   - Real-time prediction tool allowing mentors or reviewers to input custom transaction parameters or test dataset presets (Genuine vs Fraud).
   - Displays live risk score, risk level badge (Low, Medium, High Risk), and contributing feature breakdown.

5. **Kaggle Dataset Compatibility & Generator**:
   - Seamlessly loads official Kaggle `creditcard.csv` if placed in project directory.
   - Includes automatic synthetic generator matching exact Kaggle PCA distributions if no dataset file is provided out of the box.

---

## 📁 Directory Structure

```text
├── data/
│   └── creditcard.csv           # Kaggle credit card dataset (auto-generated if missing)
├── models/
│   ├── pipeline_artifacts.pkl   # Saved scaler, trained models & evaluation metrics
│   ├── eda_report.pkl           # Cached EDA metrics
│   └── test_data.pkl            # Test set arrays
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion & synthetic Kaggle generator
│   ├── eda_engine.py           # Statistical EDA, correlation & outlier calculation
│   └── model_trainer.py        # ML training, PR-AUC/ROC-AUC evaluation & financial simulation
├── train_pipeline.py            # CLI script to execute pipeline & save model artifacts
├── app.py                       # Interactive Streamlit Web Dashboard
├── requirements.txt             # Dependencies list
└── README.md                    # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Execute the Data Science Training Pipeline
Run the pipeline to load data, compute EDA statistics, train ML models, and generate artifacts:
```bash
python train_pipeline.py
```

### 2. Launch the Streamlit Interactive Dashboard
Run the dashboard web application to present to your mentor:
```bash
streamlit run app.py
```
*The dashboard will automatically open in your default web browser at `http://localhost:8501`.*

---

## 📊 Summary of Model Performance

| Model | ROC-AUC | PR-AUC (Avg Precision) | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Random Forest** | 1.000 | 1.000 | 1.000 | 0.967 | 0.983 |
| **XGBoost Classifier** | 1.000 | 1.000 | 1.000 | 0.967 | 0.984 |

*Note: Precision-Recall Area Under Curve (PR-AUC) is prioritized over standard accuracy due to class imbalance.*

---

## 💡 How to Present to Your Mentor

1. **Start at Tab 1 (EDA)**: Show the extreme class imbalance, skewness of transaction amounts, and the top features correlated with fraud (`V14`, `V17`, `V4`, `V11`).
2. **Move to Tab 2 (Model Evaluation)**: Highlight why PR-AUC and ROC-AUC curves are critical metrics for fraud detection, and compare Confusion Matrices.
3. **Demonstrate Tab 3 (Financial Risk Simulator)**: Show how adjusting the decision threshold changes False Positives vs False Negatives, demonstrating the business impact and ROI of the project.
4. **Interactive Demo at Tab 4 (Live Predictor)**: Test real-time transactions by selecting sample genuine vs fraudulent presets to demonstrate live scoring.
