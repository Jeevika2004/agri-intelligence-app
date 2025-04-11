import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load Hugging Face API token from .env
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Page config
st.set_page_config(page_title="Agri Chatbot", page_icon="💬")
st.title("🤖 Agri Intelligence Chatbot")
st.markdown("Ask me anything about crops, diseases, fertilizers, or smart farming!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Query Hugging Face Inference API
    with st.spinner("Thinking... 🤔"):
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": f"As an Agriculture Assistant:\n\n{user_input}",
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7
            }
        }

        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            # Some models return a list, some return dicts — handle both
            if isinstance(result, list):
                answer = result[0].get("generated_text", "Sorry, I couldn't generate a response.")
            elif isinstance(result, dict):
                answer = result.get("generated_text", "Sorry, I couldn't generate a response.")
            else:
                answer = "Sorry, I couldn't understand the response format."

        except Exception as e:
            answer = f"❌ Error: {str(e)}"

    # Add assistant reply to chat
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
