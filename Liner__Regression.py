import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================
# App Config
# ======================
st.set_page_config(
    page_title="Linear Regression App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Linear Regression App")
st.markdown("### Developed by Aslam Howlader")

# ======================
# Upload Data
# ======================
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("Dataset Loaded Successfully!")

# ======================
# Preview Data
# ======================
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Shape")
st.write(df.shape)

# ======================
# Convert ALL data to numeric (THE FIX)
# ======================
st.subheader("Data Processing")

# Convert every column to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for any non-numeric columns remaining
non_numeric = df.select_dtypes(include=['object']).columns.tolist()
if non_numeric:
    st.error(f"Cannot convert these columns to numbers: {non_numeric}")
    st.stop()

# Fill missing values
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

st.success(f"Processed {len(df.columns)} numeric columns")

# ======================
# Target Selection
# ======================
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(target, axis=1)
y = df[target]

# ======================
# Train Test Split
# ======================
test_size = st.slider("Test Size %", 10, 50, 20)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size/100,
    random_state=42
)

# ======================
# Scaling
# ======================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# Model Training
# ======================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ======================
# Evaluation
# ======================
st.subheader("Model Performance")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", round(mae, 4))
    st.metric("RMSE", round(rmse, 4))

with col2:
    st.metric("MSE", round(mse, 4))
    st.metric("R² Score", round(r2, 4))

# ======================
# Results Table
# ======================
st.subheader("Prediction Results")

result_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.dataframe(result_df.head(20))

# ======================
# Plot
# ======================
st.subheader("Actual vs Predicted")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)

# ======================
# Feature Importance
# ======================
st.subheader("Feature Importance")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.dataframe(coef_df)

# Plot feature importance
fig2, ax2 = plt.subplots(figsize=(10, 6))
coef_df_sorted = coef_df.sort_values('Coefficient')
ax2.barh(coef_df_sorted['Feature'], coef_df_sorted['Coefficient'])
ax2.set_xlabel("Coefficient Value")
ax2.set_title("Feature Importance")
st.pyplot(fig2)
