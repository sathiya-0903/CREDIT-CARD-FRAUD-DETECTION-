import os
import pickle
from src.data_loader import load_dataset
from src.eda_engine import generate_eda_report
from src.model_trainer import FraudDetectionPipeline

def run_pipeline():
    print("=" * 60)
    print("   CARDGUARD - EXPLAINABLE CREDIT CARD FRAUD DETECTION PIPELINE   ")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Step 1/4] Loading Dataset...")
    df_raw = load_dataset()
    print(f"Full Dataset Loaded. Total Records: {len(df_raw):,}, Features: {df_raw.shape[1]}")
    
    if len(df_raw) > 100000:
        print("[DatasetLoader] Subsampling 100,000 stratified records for optimal model training speed...")
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df_raw, train_size=100000, stratify=df_raw['Class'], random_state=42)
        df = df.reset_index(drop=True)
    else:
        df = df_raw

    # 2. Exploratory Data Analysis (EDA)
    print("\n[Step 2/4] Running Exploratory Data Analysis (EDA)...")
    eda_report = generate_eda_report(df_raw)
    class_dist = eda_report['class_distribution']
    print(f"Genuine Transactions: {class_dist['genuine_count']:,}")
    print(f"Fraud Transactions:   {class_dist['fraud_count']:,} ({class_dist['fraud_percentage']}%)")
    print(f"Imbalance Ratio:     {class_dist['imbalance_ratio']}")
    
    # 3. Model Training & Evaluation (Random Forest & XGBoost)
    print("\n[Step 3/4] Preprocessing Data & Training ML Models (Random Forest & XGBoost)...")
    pipeline = FraudDetectionPipeline(random_state=42)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    
    pipeline.train_models(X_train, y_train)
    
    print("\n[Step 4/4] Evaluating Models & Computing Metrics...")
    evaluations = pipeline.evaluate_models(X_test, y_test)
    
    print("\n" + "-" * 55)
    print(f"{'Model':<22} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1-Score':<8}")
    print("-" * 55)
    for model_name, metrics in evaluations.items():
        print(f"{model_name:<22} | {metrics['roc_auc']:<8.4f} | {metrics['pr_auc']:<8.4f} | {metrics['f1_score']:<8.4f}")
    print("-" * 55)
    
    # 4. Save Artifacts
    os.makedirs("models", exist_ok=True)
    pipeline.save_artifacts("models")
    
    with open("models/eda_report.pkl", "wb") as f:
        pickle.dump(eda_report, f)
        
    with open("models/test_data.pkl", "wb") as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)
        
    print("\n[Success] All pipeline artifacts and EDA reports saved to 'models/' directory.")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
