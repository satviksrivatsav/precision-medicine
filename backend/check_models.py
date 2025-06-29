# File: backend/check_models.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("ERROR: Google AI API Key not found in .env file.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Available models that support 'generateContent':\n")
    
    # List all available models and check which ones support the 'generateContent' method
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")