import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# App Config
# ======================
st.set_page_config(page_title="Linear Regression App", layout="wide")

st.title("📈 Linear Regression App")
st.markdown("### Developed by Aslam Howlader")

# ======================
# Upload File
# ======================
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("File Loaded Successfully")

# ======================
# Preview Data
# ======================
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Shape:", df.shape)

# ======================
# Missing Values Fix (IMPORTANT)
# ======================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

# ======================
# Encode Categorical Data
# ======================
le = LabelEncoder()

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

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
    X, y, test_size=test_size/100, random_state=42
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
st.subheader("Predictions")

result_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.dataframe(result_df.head(20))

# ======================
# Graph
# ======================
st.subheader("Actual vs Predicted")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
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
