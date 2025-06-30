# File: backend/api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import shap
import numpy as np
import re

# --- LlamaIndex Imports (from your smart chatbot) ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. App Initialization & CORS ---
app = FastAPI(title="Full AI Suite API", version="1.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 2. Transplanted Logic from Your Smart Chatbot ---

class SmartChatbot:
    # --- This class is copied directly from your app_chatbot_smart.py ---
    def __init__(self):
        self.greetings = {
            'hi': "Hi there! 👋 I'm here to help you. What would you like to know?",
            'hello': "Hello! 😊 How can I assist you?",
            'hey': "Hey! 👋 How can I help you today?",
        }
        self.thanks_responses = ["You're welcome! 😊", "Glad I could help! 👍"]
        self.goodbye_response = "Goodbye! 👋 Take care."
        self.unknown_responses = ["I'm not sure about that. I specialize in the topics from my knowledge base.", "I don't have information on that topic yet. 🤔"]
        self.similarity_threshold = 0.5
        
    def get_response_type(self, text, retrieved_nodes):
        text_lower = text.lower().strip()
        for greeting, response in self.greetings.items():
            if greeting in text_lower: return 'greeting', response
        if any(word in text_lower for word in ['thank', 'thanks']): return 'thanks', np.random.choice(self.thanks_responses)
        if any(word in text_lower for word in ['bye', 'goodbye']): return 'goodbye', self.goodbye_response
        if retrieved_nodes and retrieved_nodes[0].score >= self.similarity_threshold: return 'knowledge', None
        return 'unknown', np.random.choice(self.unknown_responses)

def parse_qa_file(file_path):
    # --- This function is copied directly from your app_chatbot_smart.py ---
    with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    qa_pairs = re.split(r'\n\s*Q:', content)
    documents = []
    for pair in qa_pairs:
        if not pair.strip(): continue
        if 'A:' in pair:
            q, a = pair.split('A:', 1)
            question = q.replace('Q:', '').strip()
            answer = a.strip()
            doc_text = f"Question: {question}\nAnswer: {answer}"
            documents.append(Document(text=doc_text, metadata={'question': question, 'answer': answer, 'file_name': os.path.basename(file_path)}))
    return documents

# --- 3. Asset Loading at Startup ---
ASSETS = {}
smart_bot = SmartChatbot()

def load_chatbot_assets():
    # --- This is the setup_pipeline logic from your smart chatbot ---
    print("--- Initializing Smart Q&A pipeline... ---")
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    try:
        context_dir = "./knowledgeBase"
        all_documents = []
        for filename in os.listdir(context_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(context_dir, filename)
                qa_docs = parse_qa_file(file_path)
                if qa_docs: all_documents.extend(qa_docs)
                else: all_documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
        index = VectorStoreIndex.from_documents(all_documents)
        print("--- Chatbot knowledge base indexed. ---")
        return index.as_retriever(similarity_top_k=3)
    except Exception as e:
        print(f"!!! ERROR building chatbot index: {e}"); return None

@app.on_event("startup")
def startup_event():
    # We don't need to load the prediction models for this final step, just the chatbot.
    print("--- Loading AI Assistant assets... ---")
    ASSETS['chatbot_retriever'] = load_chatbot_assets()
    print("--- AI Assistant ready! ---")


# --- 4. API Endpoints ---
class ChatInput(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "Smart AI Assistant API is running."}

@app.post("/chat/query")
def chat_query_endpoint(input_data: ChatInput):
    retriever = ASSETS.get('chatbot_retriever')
    if not retriever:
        return {"response_text": "Sorry, the chatbot is currently unavailable.", "type": "error"}

    question = input_data.question
    
    # Use the brain of your SmartChatbot
    retrieved_nodes = retriever.retrieve(question)
    response_type, response_text = smart_bot.get_response_type(question, retrieved_nodes)
    
    # If it's a knowledge question, get the actual text from the best node
    if response_type == 'knowledge':
        best_node = retrieved_nodes[0]
        text = best_node.get_text()
        # If it's a Q&A document, just return the answer part.
        if "Answer:" in text:
            response_text = text.split("Answer:", 1)[1].strip()
        else:
            response_text = text

    return {
        "response_text": response_text,
        "response_type": response_type
    }