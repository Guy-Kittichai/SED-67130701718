
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("best_model.pkl")

# Set background image and styling
background_image = "https://www.freecodecamp.org/news/content/images/size/w2000/2021/06/w-qjCHPZbeXCQ-unsplash.jpg"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        font-family: 'Helvetica Neue', sans-serif;  /* Modern font */
    }}
    .stTitle {{
        color: #FFD700;  /* Gold color for title */
        font-family: 'Georgia', serif;  /* Elegant font */
        font-weight: bold;
        font-size: 3em;
        margin-bottom: 40px;
    }}
    .stDetails {{
        color: #C0C0C0;  /* Silver color for details */
        font-family: 'Georgia', serif;
        font-weight: bold;
        margin-bottom: 40px;
        font-size: 1.2em;
    }}
    .stText {{
        color: #E5E5E5;  /* Soft white for text */
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.1em;
        margin-bottom: 25px;
    }}
    .stButton {{
        background-color: #333333;  /* Dark grey for button */
        color: white;
        font-weight: bold;
        font-size: 1.2em;
        padding: 5px 30px;
        border-radius: 10px;
        margin-top: 30px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);  /* Soft shadow effect */
        transition: background-color 0.3s ease;
    }}
    .stNumberInput input {{
        color: #333333;  /* Dark grey for input text */
        background-color: rgba(255, 255, 255, 0.9);
        padding: 12px;
        border-radius: 10px;
        border: 2px solid #FFD700;  /* Gold border for input fields */
        font-size: 1.1em;
    }}
    .stNumberInput input:focus {{
        outline: none;
        border-color: #FFD700;  /* Gold focus border */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("My first ML App")

# Displaying text only, no markdown or HTML
st.text("Study on Imbalanced Data Classification by 67130701718")

# Input fields
features = []
for i in range(7):  # Adjust based on dataset
    value = st.number_input(f"Feature_{i}", value=0.0)
    features.append(value)

# Prediction
if st.button("Predict"):
    prediction = model.predict([np.array(features)])
    st.write(f"Predicted Class: {prediction[0]}")
