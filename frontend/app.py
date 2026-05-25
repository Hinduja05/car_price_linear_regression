import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    layout="centered"
)

# ---------------------------------------------------
# LOAD CSS
# ---------------------------------------------------

def load_css(file_name):

    with open(file_name) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown("""
<div class="card">
    <h1>Car Price Prediction</h1>
    <p>
        Predict car prices using Linear Regression
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "processed",
    "cleaned_no_outliers.csv"
)

df = pd.read_csv(data_path)

# ---------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------

model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "linear_regression_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "scaler.pkl"
)

model = joblib.load(model_path)

scaler = joblib.load(scaler_path)

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# GRAPH 1 - PRICE DISTRIBUTION
# ---------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("Price Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    df["Price"],
    kde=True,
    ax=ax
)

ax.set_xlabel("Price")

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# GRAPH 2 - MILEAGE VS PRICE
# ---------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("Mileage vs Price")

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x=df["Mileage"],
    y=df["Price"],
    ax=ax
)

ax.set_xlabel("Mileage")
ax.set_ylabel("Price")

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# GRAPH 3 - CORRELATION HEATMAP
# ---------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8,5))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    year = st.slider(
        "Year",
        int(df["Year"].min()),
        int(df["Year"].max()),
        2020
    )

    engine_size = st.slider(
        "Engine Size",
        float(df["Engine_Size"].min()),
        float(df["Engine_Size"].max()),
        2.0
    )

    mileage = st.slider(
        "Mileage",
        int(df["Mileage"].min()),
        int(df["Mileage"].max()),
        50000
    )

with col2:

    doors = st.slider(
        "Doors",
        int(df["Doors"].min()),
        int(df["Doors"].max()),
        4
    )

    owner_count = st.slider(
        "Owner Count",
        int(df["Owner_Count"].min()),
        int(df["Owner_Count"].max()),
        1
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("Predict Price"):

    input_data = np.array([
        [
            year,
            engine_size,
            mileage,
            doors,
            owner_count
        ]
    ])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Display prediction
    st.markdown(f"""
    <div class="prediction-box">
        Predicted Car Price <br><br>
        ${prediction:,.2f}
    </div>
    """, unsafe_allow_html=True)
