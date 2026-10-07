import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

class FraudDetectionPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.evaluations = {}
        self.feature_names = []
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesses data, scales Time and Amount, and performs stratified train-test split."""
        df_copy = df.copy()
        
        # Scale Time and Amount
        df_copy['Scaled_Time'] = self.scaler.fit_transform(df_copy[['Time']])
        df_copy['Scaled_Amount'] = self.scaler.fit_transform(df_copy[['Amount']])
        
        # Drop original Time and Amount
        X = df_copy.drop(columns=['Time', 'Amount', 'Class'])
        y = df_copy['Class'].values
        
        self.feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains Random Forest and XGBoost with class-weight handling."""
        print("[ModelTrainer] Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', max_depth=12,
            random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        
        print("[ModelTrainer] Training XGBoost Classifier...")
        genuine_count = (y_train == 0).sum()
        fraud_count = (y_train == 1).sum()
        scale_pos_weight = genuine_count / max(fraud_count, 1)
        
        xgb = XGBClassifier(
            n_estimators=100, scale_pos_weight=scale_pos_weight, max_depth=6,
            learning_rate=0.1, random_state=self.random_state, eval_metric='logloss', n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        self.models['XGBoost'] = xgb

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates trained Random Forest & XGBoost models on test set."""
        eval_results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            
            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
            prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
            
            # Feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.zeros(len(self.feature_names))
                
            fi_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
            
            eval_results[name] = {
                'precision': round(float(prec), 4),
                'recall': round(float(rec), 4),
                'f1_score': round(float(f1), 4),
                'roc_auc': round(float(roc_auc), 4),
                'pr_auc': round(float(pr_auc), 4),
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                },
                'roc_curve': {'fpr': fpr_arr.tolist(), 'tpr': tpr_arr.tolist()},
                'pr_curve': {'precision': prec_arr.tolist(), 'recall': rec_arr.tolist()},
                'feature_importance': fi_df.to_dict(orient='records'),
                'y_proba': y_proba.tolist()
            }
            
        self.evaluations = eval_results
        return eval_results

    def save_artifacts(self, dirpath: str = "models"):
        """Saves scaler, trained models, feature names, and evaluation dict to disk."""
        os.makedirs(dirpath, exist_ok=True)
        artifact_data = {
            'scaler': self.scaler,
            'models': self.models,
            'evaluations': self.evaluations,
            'feature_names': self.feature_names
        }
        with open(os.path.join(dirpath, "pipeline_artifacts.pkl"), "wb") as f:
            pickle.dump(artifact_data, f)
        print(f"[ModelTrainer] Pipeline artifacts successfully saved to '{dirpath}/pipeline_artifacts.pkl'.")

    @classmethod
    def load_artifacts(cls, dirpath: str = "models"):
        """Loads pipeline artifacts from disk."""
        filepath = os.path.join(dirpath, "pipeline_artifacts.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Artifacts file not found at '{filepath}'. Please run training pipeline first.")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        pipeline = cls()
        pipeline.scaler = data['scaler']
        pipeline.models = data['models']
        pipeline.evaluations = data['evaluations']
        pipeline.feature_names = data['feature_names']
        return pipeline
