import streamlit as st
from dotenv import load_dotenv
import os
from chatbot import get_response  # Import your chatbot function

# Load variables from .env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Streamlit UI elements
st.title("Chatbot with LangChain and Google Generative AI")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input for the user
user_input = st.text_input("You:", "")


# Process user input
if user_input:
    # Get the chatbot response
    response = get_response(user_input, st.session_state.chat_history, API_KEY)
    
    # Update chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", response))

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")
