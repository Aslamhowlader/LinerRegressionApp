import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Linear Regression Aslam Howladder")

st.subheader("Machine Learning Demo")

# Sidebar
st.sidebar.header("Upload CSV Data or Use Sample")
user_example = st.sidebar.checkbox("Use example dataset")

# Load data
if user_example:
    df = sns.load_dataset("tips")
    df = df.dropna()
    st.success("Loaded sample dataset: 'tips'")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.warning("Please upload a CSV file or use example dataset")
        st.stop()

# Show data

st.write(df.head())