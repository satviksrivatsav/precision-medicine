import streamlit as st
import google.generativeai as genai
import os
import textwrap
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(page_title="Gemini-Powered Q&A Bot", page_icon="✨")

# --- Load API Key and Configure Gemini ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google AI API Key not found. Please create a .env file with your key.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('models/gemini-1.5-pro') 
except Exception as e:
    st.error(f"Failed to configure Google AI. Please check your API key. Error: {e}")
    st.stop()

# --- Context Loading (same as before) ---
@st.cache_data
def load_context(topic_filename):
    path = os.path.join("knowledgeBase", topic_filename)
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError:
        return f"Context file '{path}' not found."

# --- NEW: Function to query the Gemini API ---
@st.cache_data
def query_gemini(question, context):
    """Sends a request to the Gemini API with a specific prompt."""
    
    # This is "Prompt Engineering". We are instructing the LLM on how to behave.
    prompt = f"""
    You are an intelligent medical information assistant.
    Your task is to answer the user's question based ONLY on the provided context text.
    Do not use any external knowledge. If the answer is not found in the context, you must say "I cannot find the answer in the provided text."

    CONTEXT:
    ---
    {context}
    ---

    QUESTION: {question}

    ANSWER:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while querying the API: {e}"

# --- UI ---
st.title("✨ Gemini-Powered Medical Q&A Bot")
st.warning("⚠️ This bot uses the Google Gemini API. It is for demo purposes only and is not a substitute for professional medical advice.")

# ... (The rest of the UI is almost identical to your last chatbot app) ...
st.sidebar.title("Topic Selection")
topic = st.sidebar.selectbox("Choose a knowledge base:", ("Site Navigation", "Heart Disease", "Diabetes", "Lung Cancer", "Kidney Disease"))
topic_map = {"Site Navigation": "site_navigation.txt", "Heart Disease": "heart_disease.txt", "Diabetes": "diabetes.txt", "Lung Cancer": "lung_cancer.txt", "Kidney Disease": "kidney_disease.txt"}
context = load_context(topic_map[topic])

with st.expander(f"📚 Knowledge Base: {topic}"):
    st.markdown(f'<div style="height:200px;overflow-y:scroll;padding:10px;border:1px solid gray;">{context}</div>', unsafe_allow_html=True)
st.divider()

st.header("Ask a Question")
user_question = st.text_input("Type your question here:")

if user_question:
    if context and not context.isspace() and "not found" not in context:
        with st.spinner("✨ Asking the Gemini model..."):
            answer = query_gemini(user_question, context)
            
            st.subheader("💡 Answer")
            wrapped_text = textwrap.fill(answer, width=100)
            st.success(wrapped_text)
    else:
        st.error("The knowledge base for this topic is empty. Please add content to the corresponding .txt file.")