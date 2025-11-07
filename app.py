import streamlit as st
import pickle
import numpy as np
from PIL import Image

# ------------------------------
# Load trained model
# ------------------------------
model = pickle.load(open('Best_model_HeartD.pkl', 'rb'))

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Header Section
# ------------------------------
st.title("💖 Heart Disease Prediction App")
st.markdown("""
Welcome! This app predicts the **likelihood of heart disease** based on health parameters.  
Enter the details below and click **Predict** to know the result.
""")

st.divider()

# ------------------------------
# Sidebar Section
# ------------------------------
st.sidebar.header("🩺 Input Health Parameters")

# Example feature inputs — adjust these names and ranges based on your dataset
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=45)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=70, max_value=210, value=150)
oldpeak = st.sidebar.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0)

# Collect inputs into a numpy array
input_data = np.array([[age, sex, cp, trestbps, chol, thalach, oldpeak]])

# ------------------------------
# Prediction Section
# ------------------------------
st.subheader("🔍 Prediction Result")

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ The model predicts **Heart Disease risk is HIGH!**")
        st.markdown("Please consult a cardiologist for further checkup.")
    else:
        st.success("✅ The model predicts **No Heart Disease risk.** Stay healthy!")
        st.balloons()

st.divider()

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
---
💡 *Built with ❤️ using Streamlit and Machine Learning.*
""")
