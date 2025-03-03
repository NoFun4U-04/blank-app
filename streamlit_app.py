import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = data.target
df = df.replace({0: "setosa", 1: 'versicolor', 2: 'virginica'})

# Splitting data
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

st.title("Iris Dataset Classification using KNN")

# Display dataset
if st.checkbox("Show raw data"):
    st.write(df.head())

# Train KNN model
k_range = range(1, 26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs. K values
fig, ax = plt.subplots()
ax.plot(k_range, scores)
ax.set_xlabel("Value of K in KNN")
ax.set_ylabel("Testing Accuracy")
st.pyplot(fig)

# Select best K
best_k = scores.index(max(scores)) + 1
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

st.write(f"Best K value: {best_k}")
st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

# Prediction on random sample
if st.button("Predict Random Sample"):
    sample = df.sample(1).values
    prediction = knn.predict(sample[:, :4])[0]
    st.write(f"Predicted Species: {prediction}")
