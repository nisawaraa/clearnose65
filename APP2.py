import joblib
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model_path = r'C:\web\clearnose\clearnost_Streamlit\model (2).pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Model Prediction App",
    page_icon=":bar_chart:",
    layout="wide"
)

# Header
st.title("Clearnose")
st.write("Prediction Price Clearnose")

# Sidebar
st.sidebar.header("User Input")
st.sidebar.text("Adjust the settings below to see how the model predictions change.")

# Input Controls in Sidebar
Discount = st.sidebar.slider("Discount", min_value=0, max_value=100, value=50)
Reviews_rate = st.sidebar.slider("Reviews Rate", min_value=0, max_value=5, value=0)

# Creating a DataFrame for the input features
input_data = pd.DataFrame({
    'Discount': [Discount],
    'Reviews_rate': [Reviews_rate],
})

# Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    st.write(f"The predicted value is: {prediction[0]:.2f}")

# Display a plot for better visualization
st.subheader("Feature Distribution")

# Sample data for plotting (you should replace this with your actual data)
np.random.seed(0)  # For reproducibility
df = pd.DataFrame({
    'Discount': np.random.randint(0, 101, size=100),
    'Reviews_Rate': np.random.randint(0, 101, size=100),
})

fig, ax = plt.subplots()
sns.histplot(df['Discount'], bins=20, kde=True, ax=ax, color='pink', alpha=0.7)
ax.set_title('Distribution of Discount')
ax.set_xlabel('Discount')
ax.set_ylabel('Frequency')
st.pyplot(fig)
