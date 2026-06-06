```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="Linear Regression App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Linear Regression App")
st.markdown("### Developed by Aslam Howlader")

# ======================
# Sidebar
# ======================
st.sidebar.header("Dataset Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ======================
# Load Data
# ======================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    # -------------------
    # Dataset Preview
    # -------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------
    # Dataset Info
    # -------------------
    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    # -------------------
    # Missing Values
    # -------------------
    st.subheader("Missing Values")

    missing = df.isnull().sum()

    st.dataframe(
        missing[missing > 0]
    )

    # Fill Missing Values
    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

        else:

            df[col] = df[col].fillna(
                df[col].mean()
            )

    # -------------------
    # Encode Categorical
    # -------------------
    le = LabelEncoder()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = le.fit_transform(df[col])

    # -------------------
    # Target Selection
    # -------------------
    st.subheader("Select Target Variable")

    target = st.selectbox(
        "Target Column",
        df.columns
    )

    X = df.drop(target, axis=1)
    y = df[target]

    # -------------------
    # Test Size
    # -------------------
    test_size = st.slider(
        "Test Size %",
        10,
        50,
        20
    )

    # -------------------
    # Train Test Split
    # -------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size/100,
        random_state=42
    )

    # -------------------
    # Scaling
    # -------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------
    # Train Model
    # -------------------
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # -------------------
    # Evaluation
    # -------------------
    st.subheader("Model Evaluation")

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MAE",
            round(mae, 4)
        )

        st.metric(
            "RMSE",
            round(rmse, 4)
        )

    with col2:
        st.metric(
            "MSE",
            round(mse, 4)
        )

        st.metric(
            "R² Score",
            round(r2, 4)
        )

    # -------------------
    # Prediction Table
    # -------------------
    st.subheader("Prediction Results")

    result_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    st.dataframe(
        result_df.head(20)
    )

    # -------------------
    # Scatter Plot
    # -------------------
    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(
        figsize=(7,5)
    )

    ax.scatter(
        y_test,
        y_pred
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(
        "Actual vs Predicted"
    )

    st.pyplot(fig)

    # -------------------
    # Feature Importance
    # -------------------
    st.subheader("Feature Coefficients")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    coef_df = coef_df.sort_values(
        by="Coefficient",
        ascending=False
    )

    st.dataframe(coef_df)

else:
    st.info(
        "Please upload a CSV file to start."
    )
```
