import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Loan Dashboard", layout="wide")

st.title("💰 Loan Default Prediction System")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("processed_loan_data.csv")
df.fillna(0, inplace=True)

# Copy for UI
df_display = df.copy()

# Fix labels for dashboard
if 'Gender_Male' in df_display.columns:
    df_display['Gender'] = df_display['Gender_Male'].apply(lambda x: 'Male' if x == 1 else 'Female')

if 'Married_Yes' in df_display.columns:
    df_display['Married'] = df_display['Married_Yes'].apply(lambda x: 'Yes' if x == 1 else 'No')

# =========================
# MODEL TRAINING
# =========================
# =========================
# MODEL TRAINING (FIXED ORDER)
# =========================

# Step 1: Define X and y FIRST
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X = X.fillna(0)

# Step 2: Combine for balancing
data = X.copy()
data['Loan_Status'] = y

# Step 3: Separate classes
majority = data[data['Loan_Status'] == 0]
minority = data[data['Loan_Status'] == 1]

# Step 4: Balance dataset (IMPORTANT)
from sklearn.utils import resample

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_data = pd.concat([majority, minority_upsampled])

# Step 5: Split again
X_bal = balanced_data.drop('Loan_Status', axis=1)
y_bal = balanced_data['Loan_Status']

# Step 6: Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_bal, y_bal)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Data", "🔮 Prediction"])

# =========================
# TAB 1: DASHBOARD
# =========================
with tab1:

    st.sidebar.header("🔍 Filters")

    gender_options = ['All'] + sorted(df_display['Gender'].dropna().unique().tolist())
    married_options = ['All'] + sorted(df_display['Married'].dropna().unique().tolist())

    gender = st.sidebar.selectbox("Gender", gender_options)
    married = st.sidebar.selectbox("Married Status", married_options)

    income_range = st.sidebar.slider(
        "Applicant Income",
        int(df_display['ApplicantIncome'].min()),
        int(df_display['ApplicantIncome'].max()),
        (int(df_display['ApplicantIncome'].min()), int(df_display['ApplicantIncome'].max()))
    )

    filtered_df = df_display.copy()

    if gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == gender]

    if married != 'All':
        filtered_df = filtered_df[filtered_df['Married'] == married]

    filtered_df = filtered_df[
        (filtered_df['ApplicantIncome'] >= income_range[0]) &
        (filtered_df['ApplicantIncome'] <= income_range[1])
    ]

    if filtered_df.empty:
        st.warning("⚠️ No data available. Try different filters.")
        st.stop()

    # KPIs
    st.subheader("📊 Key Metrics")
    # -------------------------
# SMALL SIDE-BY-SIDE GRAPHS 🔥
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Status")
    fig, ax = plt.subplots(figsize=(3,2))
    sns.countplot(x='Loan_Status', data=filtered_df, palette="Set2", ax=ax)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    ax.set_title("")
    st.pyplot(fig)

with col2:
    st.subheader("Income")
    fig, ax = plt.subplots(figsize=(3,2))
    sns.histplot(filtered_df['ApplicantIncome'], kde=True, color='purple', ax=ax)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    ax.set_title("")
    st.pyplot(fig)
    

# =========================
# TAB 2: DATA
# =========================
with tab2:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df_display)

# =========================
# TAB 3: PREDICTION
# =========================
with tab3:

    st.subheader("🔮 Predict Loan Approval")

    gender_input = st.selectbox("Gender", ['Male', 'Female'])
    married_input = st.selectbox("Married", ['Yes', 'No'])
    income_input = st.number_input("Applicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)

    gender_val = 1 if gender_input == 'Male' else 0
    married_val = 1 if married_input == 'Yes' else 0

    input_data = pd.DataFrame({
        'ApplicantIncome': [income_input],
        'LoanAmount': [loan_amount],
        'Gender_Male': [gender_val],
        'Married_Yes': [married_val]
    })

    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    # ✅ EVERYTHING INSIDE THIS BLOCK
    if st.button("Predict"):

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if income_input > loan_amount * 20:
            final_result = 1
        elif income_input < loan_amount * 5:
            final_result = 0
        else:
            final_result = prediction

        st.write("Approval Probability:", round(prob*100, 2), "%")

        if final_result == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    
