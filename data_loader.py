import os
import pandas as pd
import numpy as np

def generate_synthetic_kaggle_dataset(filepath: str, n_samples: int = 50000, fraud_ratio: float = 0.003, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic Credit Card Fraud dataset matching the Kaggle Kaggle Credit Card Fraud schema:
    Columns: Time, V1-V28 (PCA components), Amount, Class (0 = Genuine, 1 = Fraud)
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_genuine = n_samples - n_fraud
    
    # Generate Time (0 to 172800 seconds ~ 48 hours)
    time_genuine = np.sort(np.random.uniform(0, 172800, n_genuine))
    time_fraud = np.sort(np.random.uniform(0, 172800, n_fraud))
    
    # Generate PCA features (V1 to V28)
    # Genuine: standard normal distribution N(0, 1)
    v_genuine = np.random.randn(n_genuine, 28)
    
    # Fraud: introduce statistically significant shifts in discriminative PCA features (V1-V28)
    # Discriminative features in Kaggle creditcard dataset: V14, V17, V12, V10, V4, V11, V3, V16
    v_fraud = np.random.randn(n_fraud, 28)
    
    # V14: Strong negative shift for fraud
    v_fraud[:, 13] = np.random.normal(-7.0, 2.5, n_fraud)
    # V17: Strong negative shift
    v_fraud[:, 16] = np.random.normal(-6.5, 2.2, n_fraud)
    # V12: Negative shift
    v_fraud[:, 11] = np.random.normal(-5.0, 2.0, n_fraud)
    # V10: Negative shift
    v_fraud[:, 9] = np.random.normal(-4.5, 2.1, n_fraud)
    # V4: Positive shift for fraud
    v_fraud[:, 3] = np.random.normal(4.5, 1.8, n_fraud)
    # V11: Positive shift
    v_fraud[:, 10] = np.random.normal(4.0, 1.7, n_fraud)
    # V3: Negative shift
    v_fraud[:, 2] = np.random.normal(-5.5, 2.0, n_fraud)
    # V16: Negative shift
    v_fraud[:, 15] = np.random.normal(-4.0, 2.0, n_fraud)

    # Generate Amount
    # Genuine: skewed lognormal (mean ~$88, median ~$22)
    amount_genuine = np.round(np.random.lognormal(mean=3.2, sigma=1.2, size=n_genuine), 2)
    # Fraud: mix of small test charges ($1-$10) and high value draining ($300-$1500)
    fraud_small = np.random.uniform(0.5, 15.0, int(n_fraud * 0.4))
    fraud_large = np.random.uniform(200.0, 1200.0, n_fraud - len(fraud_small))
    amount_fraud = np.round(np.concatenate([fraud_small, fraud_large]), 2)
    np.random.shuffle(amount_fraud)
    
    # Combine Genuine and Fraud
    df_genuine = pd.DataFrame(v_genuine, columns=[f'V{i}' for i in range(1, 29)])
    df_genuine['Time'] = time_genuine
    df_genuine['Amount'] = amount_genuine
    df_genuine['Class'] = 0
    
    df_fraud = pd.DataFrame(v_fraud, columns=[f'V{i}' for i in range(1, 29)])
    df_fraud['Time'] = time_fraud
    df_fraud['Amount'] = amount_fraud
    df_fraud['Class'] = 1
    
    df = pd.concat([df_genuine, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Order columns as Kaggle standard: Time, V1..V28, Amount, Class
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = df[cols]
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[DatasetLoader] Synthetic dataset saved to '{filepath}' with shape {df.shape} ({n_fraud} fraud cases).")
    return df


def load_dataset(filepath: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Loads dataset from filepath if present, or searches workspace for creditcard_2023.csv or creditcard.csv.
    Generates dataset if no file is found.
    """
    possible_paths = [
        "data/creditcard_2023.csv",
        filepath,
        "creditcard.csv",
        "data/creditcard.csv",
        "../creditcard.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[DatasetLoader] Loading dataset from '{path}'...")
            df = pd.read_csv(path)
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            if 'Time' not in df.columns:
                # Add default synthetic Time column if missing in 2023 dataset
                df['Time'] = np.linspace(0, 172800, len(df))
            return df
            
    print(f"[DatasetLoader] Dataset file not found. Generating synthetic dataset at '{filepath}'...")
    return generate_synthetic_kaggle_dataset(filepath)

if __name__ == "__main__":
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df['Class'].value_counts())
