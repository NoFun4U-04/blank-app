import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = data.target  # Keep numerical labels

# Splitting data
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# Mapping numerical labels to species names
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Streamlit UI
st.set_page_config(page_title="Iris Logistic Regression Classifier", layout="wide")
st.title(" Iris Dataset Classification using Logistic Regression ")
st.write("This app predicts the species of an Iris flower based on its features using Logistic Regression.")

# Manual input for prediction
st.subheader(" Predict Custom Input")
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

if st.button(" Predict Species"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = lr.predict(sample)[0]
    st.success(f"**Predicted Species:** {species_map[prediction]}")
