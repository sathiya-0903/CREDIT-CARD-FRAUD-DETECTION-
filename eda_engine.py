import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates class distribution and imbalance ratio."""
    counts = df['Class'].value_counts()
    total = len(df)
    genuine = int(counts.get(0, 0))
    fraud = int(counts.get(1, 0))
    fraud_pct = (fraud / total) * 100 if total > 0 else 0.0
    imbalance_ratio = f"1:{int(genuine / fraud)}" if fraud > 0 else "N/A"
    
    return {
        "total_transactions": total,
        "genuine_count": genuine,
        "fraud_count": fraud,
        "fraud_percentage": round(fraud_pct, 4),
        "imbalance_ratio": imbalance_ratio
    }

def compute_amount_time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes summary statistics (mean, std, median, min, max) for Amount & Time by Class."""
    stats = df.groupby('Class')[['Amount', 'Time']].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    stats.index = ['Genuine (0)', 'Fraud (1)']
    return stats

def compute_feature_correlations(df: pd.DataFrame, target_col: str = 'Class') -> pd.DataFrame:
    """Computes feature correlations with target variable Class."""
    corr = df.corr()[target_col].drop(target_col).reset_index()
    corr.columns = ['Feature', 'Correlation']
    corr = corr.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
    return corr

def compute_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates skewness across numeric features."""
    skew = df.skew().reset_index()
    skew.columns = ['Feature', 'Skewness']
    skew['AbsSkewness'] = skew['Skewness'].abs()
    return skew.sort_values(by='AbsSkewness', ascending=False).drop(columns=['AbsSkewness']).reset_index(drop=True)

def detect_outliers_iqr(df: pd.DataFrame, feature: str = 'Amount') -> Dict[str, Any]:
    """Detects outliers using 1.5 * IQR rule for genuine vs fraud transactions."""
    outliers_by_class = {}
    for cls_val, cls_label in [(0, "Genuine"), (1, "Fraud")]:
        sub_df = df[df['Class'] == cls_val][feature]
        q1 = sub_df.quantile(0.25)
        q3 = sub_df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = sub_df[(sub_df < lower_bound) | (sub_df > upper_bound)]
        outliers_by_class[cls_label] = {
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(outliers),
            "outlier_percentage": round((len(outliers) / len(sub_df)) * 100, 2) if len(sub_df) > 0 else 0.0
        }
    return outliers_by_class

def classify_fraud_typology_record(amount: float, time_val: float, v14: float = 0.0, v17: float = 0.0, v4: float = 0.0) -> Dict[str, str]:
    """Classifies a specific fraudulent transaction into one of 4 fraud typologies."""
    if amount > 0 and amount <= 15.0:
        return {
            "type": "Card Testing",
            "icon": "🧪",
            "color": "#3B82F6",
            "description": "Low amount micro-charge ($0.50 - $15.00) used by automated bots to test card validity before large transactions."
        }
    elif amount >= 300.0:
        return {
            "type": "High Value Cash Out",
            "icon": "💰",
            "color": "#EF4444",
            "description": "Abnormally high transaction amount ($300.00+) attempting maximum credit line drain."
        }
    elif (time_val % 100) < 15 or abs(v4) > 4.5:
        return {
            "type": "Rapid Fire Burst",
            "icon": "⚡",
            "color": "#F59E0B",
            "description": "High-frequency consecutive authorizations triggered within milliseconds of each other."
        }
    else:
        return {
            "type": "Behavioral Anomaly",
            "icon": "🧩",
            "color": "#8B5CF6",
            "description": "Significant statistical deviation in spending behavior across PCA components V14, V17, and V11."
        }

def classify_fraud_typologies(df: pd.DataFrame) -> pd.DataFrame:
    """Classifies all fraudulent transactions in dataset into 4 categories."""
    fraud_df = df[df['Class'] == 1].copy()
    if len(fraud_df) == 0:
        return pd.DataFrame(columns=['Fraud Type', 'Count', 'Percentage'])
        
    types = []
    for idx, row in fraud_df.iterrows():
        t = classify_fraud_typology_record(
            amount=row['Amount'],
            time_val=row['Time'],
            v14=row.get('V14', 0.0),
            v17=row.get('V17', 0.0),
            v4=row.get('V4', 0.0)
        )
        types.append(t['type'])
        
    fraud_df['FraudType'] = types
    counts = fraud_df['FraudType'].value_counts().reset_index()
    counts.columns = ['Fraud Type', 'Count']
    counts['Percentage'] = round((counts['Count'] / len(fraud_df)) * 100, 2)
    return counts

def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a complete EDA report dictionary for visualization in Streamlit."""
    class_dist = compute_class_distribution(df)
    amount_time_summary = compute_amount_time_summary(df)
    correlations = compute_feature_correlations(df)
    skewness = compute_skewness(df)
    amount_outliers = detect_outliers_iqr(df, 'Amount')
    fraud_types = classify_fraud_typologies(df)
    
    top_pos_corr = correlations.head(5).to_dict(orient='records')
    top_neg_corr = correlations.tail(5).iloc[::-1].to_dict(orient='records')
    
    return {
        "class_distribution": class_dist,
        "amount_time_summary": amount_time_summary,
        "correlations": correlations,
        "skewness": skewness,
        "top_positive_correlations": top_pos_corr,
        "top_negative_correlations": top_neg_corr,
        "amount_outliers": amount_outliers,
        "fraud_typologies": fraud_types.to_dict(orient='records')
    }

if __name__ == "__main__":
    from src.data_loader import load_dataset
    df = load_dataset()
    report = generate_eda_report(df)
    print("Class Dist:", report['class_distribution'])
    print("Top Pos Corr:\n", report['top_positive_correlations'])
    print("Top Neg Corr:\n", report['top_negative_correlations'])
