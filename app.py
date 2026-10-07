import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

from src.data_loader import load_dataset
from src.eda_engine import generate_eda_report, classify_fraud_typology_record
from src.model_trainer import FraudDetectionPipeline

st.set_page_config(
    page_title="CardGuard - Dataset & Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }
    .team-badge {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #3B82F6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .member-name {
        font-size: 1rem;
        font-weight: 700;
        color: #1E293B;
    }
    .member-sub {
        font-size: 0.85rem;
        color: #64748B;
    }
    .fraud-card {
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
    }
    .typology-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_cached_dataset():
    return load_dataset()

@st.cache_resource
def get_cached_artifacts():
    if not os.path.exists("models/pipeline_artifacts.pkl"):
        from train_pipeline import run_pipeline
        run_pipeline()
    pipeline = FraudDetectionPipeline.load_artifacts("models")
    with open("models/eda_report.pkl", "rb") as f:
        eda_report = pickle.load(f)
    with open("models/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    return pipeline, eda_report, test_data

df = get_cached_dataset()
pipeline, eda_report, test_data = get_cached_artifacts()
X_test, y_test = test_data['X_test'], test_data['y_test']

# Sidebar Navigation & Info
st.sidebar.image("https://img.icons8.com/shield-surveillance/96/000000/shield.png", width=65)
st.sidebar.title("CardGuard Panel")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Project Metadata")
st.sidebar.info(
    f"**Domain**: Finance & Security\n\n"
    f"**Dataset**: Kaggle Credit Card Fraud\n\n"
    f"**Total Records**: {len(df):,}\n\n"
    f"**SDG Alignment**: SDG 16 (Peace, Justice & Strong Institutions)"
)

st.sidebar.markdown("---")
selected_model_name = st.sidebar.radio(
    "Select Model Classifier",
    options=list(pipeline.models.keys()),
    index=0
)

# Header Section
st.markdown('<div class="main-header">🛡️ CardGuard : Explainable Fraud Analytics Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transparent Credit Card Fraud Detection • Domain: Finance • SDG 16 Alignment</div>', unsafe_allow_html=True)

# Team Information Banner
st.markdown("""
<div class="team-badge">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="card-title">Project Presenters</div>
            <div style="display: flex; gap: 2rem;">
                <div>
                    <span class="member-name">👤 Pranve.K</span> 
                    <span class="member-sub">(Regd No.: 210425243188 • Sec E)</span>
                </div>
                <div>
                    <span class="member-name">👤 Sathyanarayanan.G</span> 
                    <span class="member-sub">(Regd No.: 210425243253 • Sec E)</span>
                </div>
            </div>
        </div>
        <div>
            <span style="background-color: #DBEAFE; color: #1E40AF; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                Review 2 Completed
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 1. Project Progress & Review Roadmap",
    "📁 2. Dataset Explorer & Data Table",
    "🚨 3. Fraud Typologies & EDA",
    "🤖 4. Model Evaluation & Comparison",
    "🔮 5. Live Fraud Risk & Pattern Detector"
])

# ==============================================================================
# TAB 1: REVIEW PROCESS & PROJECT ROADMAP
# ==============================================================================
with tab1:
    st.subheader("1. 12-Week Methodology & Review Progress")
    st.write("Tracking project milestones, data cleaning progress, injected missing value treatments, and innovation highlights.")
    
    # 12-Week Interactive Timeline Progress
    roadmap_data = pd.DataFrame([
        {"Milestone": "Wk 1-2: Problem Definition & Literature Survey", "Progress": 100, "Status": "Completed"},
        {"Milestone": "Wk 2-3: Dataset Collection (Kaggle Dataset)", "Progress": 100, "Status": "Completed"},
        {"Milestone": "Wk 3-4: Exploratory Data Analysis (EDA)", "Progress": 100, "Status": "Completed"},
        {"Milestone": "Wk 4-5: Preprocessing & Imputation (Mean/Median)", "Progress": 100, "Status": "Completed"},
        {"Milestone": "Wk 5-7: Feature Analysis & Selection (V4, V11, V14, V17)", "Progress": 100, "Status": "Completed"},
        {"Milestone": "Wk 7-9: Model Optimization (Random Forest & XGBoost)", "Progress": 100, "Status": "Completed"},
        {"Milestone": "Wk 9-10: Performance Evaluation (ROC-AUC & PR-AUC)", "Progress": 100, "Status": "Completed"},
        {"Milestone": "Wk 11-12: Final Mentor Presentation & Dashboard", "Progress": 95, "Status": "In Review"}
    ])
    
    fig_progress = px.bar(
        roadmap_data, x="Progress", y="Milestone", orientation="h",
        color="Progress", color_continuous_scale="Viridis",
        text="Status", title="12-Week Project Roadmap Execution Progress (%)"
    )
    fig_progress.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=30, b=20, l=20, r=20))
    st.plotly_chart(fig_progress, width="stretch")
    
    st.markdown("---")
    
    col_proc1, col_proc2 = st.columns(2)
    
    with col_proc1:
        st.markdown("##### 🛠️ Data Preprocessing: Missing Value Injection & Treatment")
        st.write("""
        To simulate real-world data gaps during review analysis, missing values were injected across features:
        - **720 Injected Gaps**: Across `Amount`, `V1`, and `V4` (~6% missing rate).
        - **Imputation Strategy**:
          - `V1` & `V4` (Symmetric distribution): Imputed using **Column Mean**.
          - `Amount` (Right-skewed distribution): Imputed using **Column Median** (robust to extreme outliers).
        - **Result**: **0 missing values remain** — dataset fully populated.
        """)
        
        missing_df = pd.DataFrame([
            {"Stage": "Raw Missing Values Injected", "Count": 720},
            {"Stage": "After Mean Imputation (V1, V4)", "Count": 240},
            {"Stage": "After Median Imputation (Amount)", "Count": 0}
        ])
        fig_miss = px.funnel(missing_df, x="Count", y="Stage", color="Stage", title="Missing Value Treatment Progress")
        st.plotly_chart(fig_miss, width="stretch")
        
    with col_proc2:
        st.markdown("##### 📐 Outlier Treatment: IQR Winsorization")
        st.write("""
        - **Detection Rule**: Extreme values identified using $1.5 \\times \\text{IQR}$ beyond $Q_1$ and $Q_3$ fences on `Amount`.
        - **Treatment Method**: **IQR Winsorization** (Capping extreme values at IQR upper/lower fences rather than deleting rows).
        - **Impact**: Preserves total sample size while preventing extreme skewness from corrupting decision boundaries.
        """)
        
        outlier_data = pd.DataFrame([
            {"Metric": "Raw Outlier Count", "Value": eda_report['amount_outliers']['Genuine']['outlier_count']},
            {"Metric": "Upper Fence Cap ($)", "Value": float(eda_report['amount_outliers']['Genuine']['upper_bound'])},
            {"Metric": "Lower Fence Cap ($)", "Value": float(eda_report['amount_outliers']['Genuine']['lower_bound'])}
        ])
        st.dataframe(outlier_data, width="stretch")

# ==============================================================================
# TAB 2: DATASET EXPLORER & DATA TABLE (NEW)
# ==============================================================================
with tab2:
    st.subheader("2. Kaggle Credit Card Dataset Explorer")
    st.write("Inspect, filter, search, and export the raw credit card transaction dataset.")
    
    # Dataset Filtering Controls
    fcol1, fcol2, fcol3 = st.columns(3)
    
    with fcol1:
        class_filter = st.selectbox(
            "Filter by Class Label:",
            options=["All Transactions", "Genuine Transactions (Class 0)", "Fraudulent Transactions (Class 1)"]
        )
    with fcol2:
        min_amt, max_amt = float(df['Amount'].min()), float(df['Amount'].max())
        amount_range = st.slider("Filter by Transaction Amount ($):", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt))
    with fcol3:
        rows_to_display = st.selectbox("Rows to Display:", options=[100, 500, 1000, 5000, 10000, "All Rows"], index=0)
        
    # Apply Filters
    df_filtered = df.copy()
    if class_filter == "Genuine Transactions (Class 0)":
        df_filtered = df_filtered[df_filtered['Class'] == 0]
    elif class_filter == "Fraudulent Transactions (Class 1)":
        df_filtered = df_filtered[df_filtered['Class'] == 1]
        
    df_filtered = df_filtered[(df_filtered['Amount'] >= amount_range[0]) & (df_filtered['Amount'] <= amount_range[1])]
    
    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records in View", f"{len(df_filtered):,}")
    m2.metric("Genuine Count in View", f"{(df_filtered['Class'] == 0).sum():,}")
    m3.metric("Fraud Count in View", f"{(df_filtered['Class'] == 1).sum():,}")
    m4.metric("Avg Amount in View", f"${df_filtered['Amount'].mean():,.2f}")
    
    st.markdown("---")
    st.markdown("##### Interactive Dataset Table")
    
    display_df = df_filtered if rows_to_display == "All Rows" else df_filtered.head(rows_to_display)
    st.dataframe(display_df, width="stretch", height=450)
    
    # CSV Download Button
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset as CSV",
        data=csv_bytes,
        file_name="cardguard_filtered_transactions.csv",
        mime="text/csv"
    )

# ==============================================================================
# TAB 3: FRAUD TYPOLOGIES & EDA
# ==============================================================================
with tab3:
    st.subheader("3. Detected Fraud Typologies & Exploratory Analytics")
    st.write("CardGuard categorizes all fraudulent activity into 4 distinct operational typologies:")
    
    # 4 Fraud Typology Cards
    ft1, ft2, ft3, ft4 = st.columns(4)
    with ft1:
        st.markdown("""
        <div class="fraud-card" style="background-color: #FEF3C7; border-color: #FCD34D;">
            <div class="typology-badge" style="background: #F59E0B; color: white;">⚡ Rapid Fire Burst</div>
            <h4 style="margin: 0.2rem 0; color: #78350F;">High-Frequency Burst</h4>
            <p style="color: #92400E; font-size: 0.85rem;">Multiple consecutive authorizations triggered within milliseconds of each other by automated scripts.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with ft2:
        st.markdown("""
        <div class="fraud-card" style="background-color: #EFF6FF; border-color: #BFDBFE;">
            <div class="typology-badge" style="background: #3B82F6; color: white;">🧪 Card Testing</div>
            <h4 style="margin: 0.2rem 0; color: #1E40AF;">Micro-Charge Bots</h4>
            <p style="color: #1E3A8A; font-size: 0.85rem;">Small transactions ($0.50 - $15.00) executed to test if stolen card credentials are active.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with ft3:
        st.markdown("""
        <div class="fraud-card" style="background-color: #FEE2E2; border-color: #FCA5A5;">
            <div class="typology-badge" style="background: #EF4444; color: white;">💰 High Value Cash Out</div>
            <h4 style="margin: 0.2rem 0; color: #991B1B;">Credit Limit Drain</h4>
            <p style="color: #7F1D1D; font-size: 0.85rem;">Abnormally high transaction amounts ($300.00+) attempting to maximize single-event theft.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with ft4:
        st.markdown("""
        <div class="fraud-card" style="background-color: #F3E8FF; border-color: #DDD6FE;">
            <div class="typology-badge" style="background: #8B5CF6; color: white;">🧩 Behavioral Anomaly</div>
            <h4 style="margin: 0.2rem 0; color: #5B21B6;">Pattern Deviation</h4>
            <p style="color: #4C1D95; font-size: 0.85rem;">Significant deviation in spending patterns across component features V14, V17, and V11.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Dataset Breakdown of Fraud Typologies
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("##### Fraud Typology Distribution across Dataset")
        ft_df = pd.DataFrame(eda_report['fraud_typologies'])
        fig_ft = px.pie(
            ft_df, values='Count', names='Fraud Type',
            color='Fraud Type',
            color_discrete_map={
                'Rapid Fire Burst': '#F59E0B',
                'Card Testing': '#3B82F6',
                'High Value Cash Out': '#EF4444',
                'Behavioral Anomaly': '#8B5CF6'
            },
            hole=0.45, title="Proportion of Fraud Types Detected"
        )
        st.plotly_chart(fig_ft, width="stretch")
        
    with f_col2:
        st.markdown("##### Fraud Typology Counts & Percentages")
        st.dataframe(ft_df, width="stretch")
        
    st.markdown("---")
    st.markdown("##### General Dataset Class Distribution & Amount Analysis")
    
    cd = eda_report['class_distribution']
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(
            values=[cd['genuine_count'], cd['fraud_count']],
            names=['Genuine (0)', 'Fraud (1)'],
            color_discrete_sequence=['#10B981', '#EF4444'],
            hole=0.45, title="Class Distribution (Genuine vs Fraud)"
        )
        st.plotly_chart(fig_pie, width="stretch")
        
    with c2:
        fig_amt = px.histogram(
            df, x="Amount", color="Class",
            log_y=True, nbins=50,
            color_discrete_map={0: '#10B981', 1: '#EF4444'},
            barmode="overlay", opacity=0.75,
            title="Amount Distribution (Log Scale)"
        )
        st.plotly_chart(fig_amt, width="stretch")

# ==============================================================================
# TAB 4: MODEL EVALUATION & COMPARISON (RF & XGBOOST)
# ==============================================================================
with tab4:
    st.subheader("4. Model Benchmarking & Evaluation (Random Forest vs XGBoost)")
    st.write("Evaluating Random Forest Classifier and XGBoost Classifier using metrics suited for fraud prediction.")
    
    # Summary Table
    comp_records = []
    for model_name, evals in pipeline.evaluations.items():
        cm = evals['confusion_matrix']
        comp_records.append({
            "Model Name": model_name,
            "ROC-AUC": evals['roc_auc'],
            "PR-AUC": evals['pr_auc'],
            "Precision": evals['precision'],
            "Recall": evals['recall'],
            "F1-Score": evals['f1_score'],
            "True Positives (TP)": cm['tp'],
            "False Positives (FP)": cm['fp'],
            "False Negatives (FN)": cm['fn'],
            "True Negatives (TN)": cm['tn']
        })
        
    comp_df = pd.DataFrame(comp_records).set_index("Model Name")
    st.dataframe(comp_df, width="stretch")
    
    st.markdown("---")
    c_curve1, c_curve2 = st.columns(2)
    
    with c_curve1:
        st.markdown("##### Receiver Operating Characteristic (ROC Curves)")
        fig_roc = go.Figure()
        for model_name, evals in pipeline.evaluations.items():
            fig_roc.add_trace(go.Scatter(
                x=evals['roc_curve']['fpr'],
                y=evals['roc_curve']['tpr'],
                mode='lines',
                name=f"{model_name} (AUC = {evals['roc_auc']:.4f})"
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Classifier'))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_roc, width="stretch")
        
    with c_curve2:
        st.markdown("##### Precision-Recall (PR) Curves")
        fig_pr = go.Figure()
        for model_name, evals in pipeline.evaluations.items():
            fig_pr.add_trace(go.Scatter(
                x=evals['pr_curve']['recall'],
                y=evals['pr_curve']['precision'],
                mode='lines',
                name=f"{model_name} (PR-AUC = {evals['pr_auc']:.4f})"
            ))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_pr, width="stretch")
        
    st.markdown("---")
    st.markdown("##### Confusion Matrix Heatmaps")
    m_cols = st.columns(2)
    for idx, (model_name, evals) in enumerate(pipeline.evaluations.items()):
        with m_cols[idx]:
            cm = evals['confusion_matrix']
            cm_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
            fig_cm = px.imshow(
                cm_arr, text_auto=True,
                x=['Pred Genuine (0)', 'Pred Fraud (1)'],
                y=['Actual Genuine (0)', 'Actual Fraud (1)'],
                color_continuous_scale='Blues',
                title=f"{model_name} Confusion Matrix"
            )
            fig_cm.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_cm, width="stretch")

# ==============================================================================
# TAB 5: LIVE FRAUD RISK & PATTERN DETECTOR
# ==============================================================================
with tab5:
    st.subheader("5. Real-Time Fraud Risk & Typology Detector")
    st.write("Test individual transactions. Choose a preset or customize parameters below:")
    
    preset = st.radio(
        "Select Test Preset / Fraud Type to Simulate:",
        [
            "Genuine Transaction Sample",
            "Card Testing Simulation ($2.50 Micro-Charge)",
            "High Value Cash Out Simulation ($850.00 Drain)",
            "Rapid Fire Burst Simulation",
            "Behavioral Anomaly Simulation"
        ],
        horizontal=True
    )
    
    fraud_indices = np.where(y_test == 1)[0]
    genuine_indices = np.where(y_test == 0)[0]
    
    if preset == "Genuine Transaction Sample":
        sample_idx = genuine_indices[0] if len(genuine_indices) > 0 else 0
        init_sample = X_test[sample_idx]
        init_amount = float(df[df['Class'] == 0]['Amount'].iloc[0])
        init_time = float(df[df['Class'] == 0]['Time'].iloc[0])
    elif "Card Testing" in preset:
        sample_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
        init_sample = X_test[sample_idx]
        init_amount = 2.50
        init_time = 1200.0
    elif "High Value Cash Out" in preset:
        sample_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
        init_sample = X_test[sample_idx]
        init_amount = 850.00
        init_time = 45000.0
    elif "Rapid Fire Burst" in preset:
        sample_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
        init_sample = X_test[sample_idx]
        init_amount = 120.0
        init_time = 1005.0
    else:
        sample_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
        init_sample = X_test[sample_idx]
        init_amount = 195.0
        init_time = 85000.0

    st.markdown("##### Transaction Input Parameters")
    ic1, ic2, ic3, ic4 = st.columns(4)
    amount_input = ic1.number_input("Transaction Amount ($)", value=init_amount, step=10.0)
    time_input = ic2.number_input("Time Elapsed (s)", value=init_time, step=1000.0)
    v14_input = ic3.number_input("V14 (Discriminative Feature)", value=float(init_sample[pipeline.feature_names.index('V14')]) if 'V14' in pipeline.feature_names else 0.0, step=0.5)
    v17_input = ic4.number_input("V17 (Discriminative Feature)", value=float(init_sample[pipeline.feature_names.index('V17')]) if 'V17' in pipeline.feature_names else 0.0, step=0.5)
    
    with st.expander("Expand to Edit All PCA Features (V1 - V28)"):
        adv_cols = st.columns(7)
        feat_values = list(init_sample)
        for i in range(1, 29):
            feat_name = f'V{i}'
            if feat_name in pipeline.feature_names:
                idx = pipeline.feature_names.index(feat_name)
                if feat_name not in ['V14', 'V17']:
                    val = adv_cols[(i-1) % 7].number_input(feat_name, value=float(feat_values[idx]), step=0.5, key=f"inp_{feat_name}")
                    feat_values[idx] = val
        feat_values[pipeline.feature_names.index('V14')] = v14_input
        feat_values[pipeline.feature_names.index('V17')] = v17_input
        
    scaled_time = float(pipeline.scaler.transform([[time_input]])[0][0])
    scaled_amount = float(pipeline.scaler.transform([[amount_input]])[0][0])
    
    feat_values[pipeline.feature_names.index('Scaled_Time')] = scaled_time
    feat_values[pipeline.feature_names.index('Scaled_Amount')] = scaled_amount
    
    input_arr = np.array(feat_values).reshape(1, -1)
    
    st.markdown("---")
    model = pipeline.models[selected_model_name]
    fraud_prob = float(model.predict_proba(input_arr)[0, 1])
    
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        st.markdown(f"##### Risk Assessment ({selected_model_name})")
        st.metric("Fraud Probability Score", f"{fraud_prob * 100:.2f}%")
        
        if fraud_prob < 0.30:
            st.success("🟢 LOW RISK - Legitimate Transaction Approved")
        elif fraud_prob < 0.70:
            st.warning("🟡 MEDIUM RISK - Marked for Verification")
        else:
            st.error("🔴 HIGH RISK - FRAUDULENT TRANSACTION FLAGGED!")
            
            ft_res = classify_fraud_typology_record(
                amount=amount_input,
                time_val=time_input,
                v14=v14_input,
                v17=v17_input,
                v4=feat_values[pipeline.feature_names.index('V4')] if 'V4' in pipeline.feature_names else 0.0
            )
            
            st.markdown(f"""
            <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #991B1B; text-transform: uppercase;">Detected Fraud Pattern</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #7F1D1D; margin: 0.2rem 0;">
                    {ft_res['icon']} {ft_res['type']}
                </div>
                <div style="font-size: 0.85rem; color: #B91C1C;">
                    {ft_res['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with p_col2:
        st.markdown("##### Feature Values vs Baseline")
        top_contrib_df = pd.DataFrame({
            'Feature': ['V14', 'V17', 'V12', 'V10', 'V4', 'V11', 'Scaled_Amount'],
            'Value': [
                v14_input, v17_input,
                feat_values[pipeline.feature_names.index('V12')],
                feat_values[pipeline.feature_names.index('V10')],
                feat_values[pipeline.feature_names.index('V4')],
                feat_values[pipeline.feature_names.index('V11')],
                scaled_amount
            ]
        })
        fig_contrib = px.bar(top_contrib_df, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='PuOr')
        fig_contrib.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_contrib, width="stretch")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748B;'>CardGuard • Pranve.K & Sathyanarayanan.G • Department of Finance & Data Analytics</div>", unsafe_allow_html=True)
