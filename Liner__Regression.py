import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config(
    page_title="Linear Regression App",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1 style="color:white;">📈 Linear Regression App</h1></div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Developed by <b>Aslam Howlader</b></p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload your CSV file",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel file with numeric data"
)

if uploaded_file is None:
    st.info("👈 Please upload a CSV file to get started!")
    st.markdown("""
    ### How to use this app:
    1. **Upload** your dataset (CSV format)
    2. **Select** the target column you want to predict
    3. **Adjust** the test size percentage
    4. **View** the model performance and visualizations
    
    ### Requirements:
    - Your data should contain numeric values
    - First row should be column headers
    - Target column should be numeric
    """)
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

df = load_data(uploaded_file)

# Display success
st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# Show raw data toggle
if st.checkbox("🔍 Show raw data"):
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(20))
    st.caption(f"Total rows: {len(df)} | Total columns: {len(df.columns)}")

# Data preprocessing
st.subheader("🔄 Data Preprocessing")

# Show data types before conversion
col1, col2 = st.columns(2)
with col1:
    st.write("**Original Data Types:**")
    st.write(df.dtypes)

# Convert everything to numeric
df_numeric = df.copy()
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

with col2:
    st.write("**Converted Data Types:**")
    st.write(df_numeric.dtypes)

# Show missing values
missing_counts = df_numeric.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]

if len(missing_cols) > 0:
    st.warning(f"⚠️ Found {len(missing_cols)} columns with missing values")
    
    # Fill missing values
    for col in df_numeric.columns:
        if df_numeric[col].isnull().any():
            df_numeric[col] = df_numeric[col].fillna(df_numeric[col].mean())
    
    st.success("✅ Missing values filled with column means")
else:
    st.success("✅ No missing values found!")

# Remove any remaining non-numeric columns
numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()
df_clean = df_numeric[numeric_cols]

if len(df_clean.columns) < len(df.columns):
    removed = set(df.columns) - set(numeric_cols)
    st.info(f"ℹ️ Removed non-numeric columns: {removed}")

# Target selection
st.subheader("🎯 Model Configuration")
target = st.selectbox(
    "Select target column (what you want to predict)",
    df_clean.columns,
    help="This will be your dependent variable Y"
)

# Features
features = [col for col in df_clean.columns if col != target]

if len(features) == 0:
    st.error("❌ No feature columns available! Please check your data.")
    st.stop()

st.write(f"**Features:** {len(features)} columns")

# Option to select features
feature_selection = st.radio(
    "Feature selection:",
    ["Use all features", "Select specific features"],
    horizontal=True
)

if feature_selection == "Select specific features":
    selected_features = st.multiselect(
        "Choose features for prediction:",
        features,
        default=features[:min(5, len(features))]
    )
    if not selected_features:
        st.error("Please select at least one feature")
        st.stop()
    features = selected_features

# Prepare data
X = df_clean[features]
y = df_clean[target]

# Train-test split
test_size = st.slider(
    "Test set size (%):",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
    help="Percentage of data to use for testing"
) / 100

# Random state for reproducibility
random_state = st.number_input("Random state:", value=42, min_value=0, max_value=100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

st.info(f"📊 Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# Scaling option
use_scaling = st.checkbox("Apply feature scaling (StandardScaler)", value=True)

if use_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Train model
with st.spinner("Training model..."):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Model evaluation
st.subheader("📊 Model Performance")

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error")
with metric_cols[1]:
    st.metric("MSE", f"{mse:.4f}", help="Mean Squared Error")
with metric_cols[2]:
    st.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error")
with metric_cols[3]:
    st.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination")

# Interpretation
if r2 > 0.8:
    st.success(f"✅ Excellent model! R² = {r2:.4f} (80%+ variance explained)")
elif r2 > 0.6:
    st.info(f"👍 Good model! R² = {r2:.4f} (60-80% variance explained)")
elif r2 > 0.4:
    st.warning(f"⚠️ Moderate model! R² = {r2:.4f} (40-60% variance explained)")
else:
    st.error(f"❌ Poor model! R² = {r2:.4f} (<40% variance explained)")

# Visualizations
st.subheader("📈 Visualizations")

viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance"])

with viz_tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel("Actual Values", fontsize=12)
    ax1.set_ylabel("Predicted Values", fontsize=12)
    ax1.set_title("Actual vs Predicted Values", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with viz_tab2:
    residuals = y_test - y_pred
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residual plot
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel("Predicted Values", fontsize=12)
    ax2.set_ylabel("Residuals", fontsize=12)
    ax2.set_title("Residual Plot", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Residuals", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title("Distribution of Residuals", fontsize=14, fontweight='bold')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)

with viz_tab3:
    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=True)
    
    fig3, ax4 = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
    colors = ['red' if x < 0 else 'green' for x in coef_df['Coefficient']]
    ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    ax4.set_xlabel("Coefficient Value", fontsize=12)
    ax4.set_title("Feature Importance", fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig3)
    
    # Display coefficients table
    st.dataframe(coef_df.sort_values('Coefficient', ascending=False), use_container_width=True)

# Prediction results
st.subheader("📋 Prediction Results")
show_rows = st.slider("Number of rows to display:", 5, 50, 10)

results_df = pd.DataFrame({
    "Actual": y_test.values[:show_rows],
    "Predicted": y_pred[:show_rows],
    "Error": (y_test.values[:show_rows] - y_pred[:show_rows])
})
results_df["Absolute Error"] = np.abs(results_df["Error"])
results_df["Error %"] = (np.abs(results_df["Error"] / results_df["Actual"]) * 100).round(2)

st.dataframe(results_df, use_container_width=True)

# Model equation
st.subheader("📐 Model Equation")
equation = f"{target} = {model.intercept_:.4f}"
for feat, coef in zip(features, model.coef_):
    equation += f" + ({coef:.4f} × {feat})"
st.code(equation, language="python")

# Download results
st.subheader("💾 Download Results")
csv = results_df.to_csv(index=False)
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name="predictions.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Linear Regression App | Built with Streamlit</p>",
    unsafe_allow_html=True
)
