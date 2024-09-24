import streamlit as st
import pickle
import pandas as pd
import numpy as np

model_path  = 'model.pkl'
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
st.write("Prediction Price Clearnose ")


# Sidebar
# Sidebar
st.sidebar.header("User Input")
st.sidebar.text("Adjust the settings below to see how the model predictions change.")

# Input Controls in Sidebar
feature2 = st.sidebar.slider("Discount", min_value=0, max_value=100, value=50)
feature3 = st.sidebar.slider("Reviews_rate", min_value=0, max_value=100, value=50)


# Creating a DataFrame for the input features
input_data = pd.DataFrame({
    'Feature2': [feature2],
    'Feature3': [feature3],
})


# Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    st.write(f"The predicted value is: {prediction[0]:.2f}")

# Display a plot for better visualization
st.subheader("Feature Distribution")

# Generating sample data for plotting (replace with your actual data if available)
data = {
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100) * 100
}
df = pd.DataFrame(data)

fig, ax = plt.subplots()
sns.histplot(df['Feature1'], bins=20, kde=True, ax=ax, color='pink', alpha=0.7)
ax.set_title('Distribution of Feature 1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Frequency')

st.pyplot(fig)
