import streamlit as st
import numpy as np
import os
from keras.models import load_model  # type: ignore
from keras.preprocessing import image  # type: ignore
from PIL import Image
import joblib
import requests
import openai
from dotenv import load_dotenv
from utils import get_weather
from firebase_utils import init_firebase, save_chat, save_crop_prediction, save_leaf_disease

# ---------------- Initialize ----------------
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
db = init_firebase()

st.set_page_config(page_title="Agri Intelligence", layout="wide")

# ---------------- Utility Function ----------------
def get_disease_treatment(disease_name):
    prompt = f"As an Agriculture Expert, provide short and clear treatment steps for {disease_name} in crops."

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.6
        }
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            headers=headers,
            json=payload,
        )

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        else:
            return "⚠️ Unable to retrieve treatment steps. Try again later."
    except Exception as e:
        return f"⚠️ Error fetching treatment info: {str(e)}"

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("🌿 AGRI Assistant")
page = st.sidebar.radio("Go to", ["🏠 Home", "🌾 Crop Recommendation", "🍃 Leaf Disease Detection", "💬 Agri Chatbot"])

# ---------------- Custom Styling ----------------
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #f9f7f1;
        }
        .stButton>button {
            border-radius: 10px;
            background-color: #6ca965;
            color: white;
            padding: 0.5em 2em;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #4e854a;
        }
        .main-card {
            background-color: #ffffff;
            padding: 2em;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 2em;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------- Home Page ----------------
if page == "🏠 Home":
    st.markdown("""
        <h1 style='text-align: center;'>🌱 Agri Intelligence Assistant</h1>
        <p style='text-align: center; font-size: 1.2em;'>Your AI-powered partner for smarter agriculture – recommend crops, detect leaf diseases, and chat with an agri-expert bot!</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🌾 Crop Recommendation")
        st.write("Input soil nutrients & weather to get the best crop for your land.")
    with col2:
        st.subheader("🍃 Leaf Disease Detection")
        st.write("Upload a plant leaf image to detect possible diseases using AI.")
    with col3:
        st.subheader("💬 Agri Chatbot")
        st.write("Ask your questions about farming, fertilizers, crops, and more.")

    st.markdown("""
        <hr>
        <p style='text-align: center;'>🚜 Developed with ❤️ for Smart Agriculture | Version 1.0</p>
    """, unsafe_allow_html=True)

# ---------------- Crop Recommendation ----------------
elif page == "🌾 Crop Recommendation":
    st.title("🌾 Crop Recommendation System")

    model_crop = joblib.load("crop_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    with st.form("crop_form"):
        city = st.text_input("Enter your City")
        N = st.slider("Nitrogen (N)", 0, 200, 50)
        P = st.slider("Phosphorous (P)", 0, 200, 50)
        K = st.slider("Potassium (K)", 0, 200, 50)
        ph = st.slider("pH value", 0.0, 14.0, 6.5)
        rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
        submitted = st.form_submit_button("Predict Crop")

    if submitted:
        temperature, humidity = get_weather(city)

        if temperature is None or humidity is None:
            st.error("❌ Failed to fetch weather. Check city name or API key.")
        else:
            st.markdown("<div class='main-card'>", unsafe_allow_html=True)
            st.info(f"🌡️ Temperature: {temperature} °C | 💧 Humidity: {humidity}%")

            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            pred_encoded = model_crop.predict(features)[0]
            crop = label_encoder.inverse_transform([pred_encoded])[0]
            st.success(f"✅ Recommended Crop: 🌾 **{crop.upper()}**")
            st.markdown("</div>", unsafe_allow_html=True)

            save_crop_prediction(db, city, N, P, K, temperature, humidity, ph, rainfall, crop)

# ---------------- Leaf Disease Detection ----------------
elif page == "🍃 Leaf Disease Detection":
    st.title("🍃 Leaf Disease Detector")

    model_leaf = load_model("leaf_disease_model.h5")
    data_dir = "data/archive/plantvillage dataset/color"
    class_labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).resize((128, 128))
        st.image(img, caption="📸 Uploaded Leaf Image", use_column_width=True)

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Detect Disease"):
            prediction = model_leaf.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            st.markdown("<div class='main-card'>", unsafe_allow_html=True)
            st.success(f"🔍 Predicted Disease: **{predicted_class}**")

            with st.spinner("Fetching treatment suggestions..."):
                treatment = get_disease_treatment(predicted_class)

            st.markdown("#### 💊 Suggested Treatment:")
            st.markdown(f"{treatment}")
            st.markdown("</div>", unsafe_allow_html=True)

            save_leaf_disease(db, uploaded_file.name, predicted_class)

# ---------------- Agri Chatbot ----------------
elif page == "💬 Agri Chatbot":
    st.title("🤖 Agri Intelligence Chatbot")
    st.markdown("Ask me anything about crops, diseases, fertilizers, or smart farming!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Type your question here...")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            headers = {
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": f"As an Agriculture Assistant:\n\n{user_input}",
                "parameters": {"max_new_tokens": 200, "temperature": 0.7},
            }

            try:
                response = requests.post(
                    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                    headers=headers,
                    json=payload,
                )
                result = response.json()
                answer = result[0].get("generated_text", "Sorry, I couldn't generate a response.")
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        save_chat(db, user="guest_user", question=user_input, response=answer)
