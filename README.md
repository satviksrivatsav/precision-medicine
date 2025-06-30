# 🔬 Precision Medicine & AI Assistant Suite

This repository contains the complete backend for a multi-functional healthcare AI platform. The backend is built as a robust FastAPI service that provides:
1.  **Four Disease Prediction Models:**
    *   Heart Disease Mortality
    *   Diabetes Risk
    *   Lung Cancer Survival
    *   Chronic Kidney Disease
2.  **An AI Assistant Chatbot:** A local, private Q&A bot powered by a custom NLP pipeline to answer questions about medical topics and site navigation from a provided knowledge base.

The backend is designed as a headless API, intended to be used by a separate frontend application.

---

## 🚀 Backend Setup and Execution

Follow these instructions to get the backend API service running on your local machine.

### 1. Prerequisites

*   Python 3.10+
*   Git

### 2. Setup

1.  **Clone the Repository:**
    ```sh
    git clone https://github.com/satviksrivatsav/precision-medicine.git
    cd precision-medicine
    ```

2.  **Create and Activate a Virtual Environment:**
    This isolates the project's dependencies.
    *   On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    This command installs all necessary libraries, including FastAPI, Scikit-learn, and LlamaIndex.
    ```sh
    pip install -r requirements.txt
    ```

### 3. Running the API Server

1.  **Navigate to the `backend` directory:**
    ```sh
    cd backend
    ```

2.  **Start the Uvicorn Server:**
    This command starts the live API server. The `--reload` flag automatically restarts the server if you make any code changes.
    ```sh
    uvicorn api:app --reload
    ```

3.  **Confirmation:**
    Once the server starts, you will see output in your terminal indicating that the application is running, typically on `http://127.0.0.1:8000`. You will also see logs as the AI models and chatbot index are loaded into memory.

    ```
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Started reloader process [12345]...
    INFO:     Started server process [67890]
    INFO:     Waiting for application startup.
    --- Loading all model assets... ---
    --- Chatbot knowledge base indexed. ---
    --- All assets loaded! ---
    INFO:     Application startup complete.
    ```

### 4. Verifying the API

You can verify that the API is working correctly in two ways:

1.  **Root Endpoint:** Open your web browser and navigate to `http://127.0.0.1:8000`. You should see a JSON response like:
    `{"status":"AI Health Suite API is running."}`

2.  **Interactive Documentation (Swagger UI):** This is the most important tool for frontend development. Navigate to `http://127.0.0.1:8000/docs`. You will find a complete, interactive documentation page where you can see all available endpoints, their required inputs, and even test them live from your browser.

---

##  Frontend Integration Guide

This section is for the frontend development team.

### API Base URL

When running the backend locally, the base URL for all API calls is:
`http://127.0.0.1:8000`

### Available Endpoints

Please refer to the interactive documentation at `http://127.0.0.1:8000/docs` for detailed schemas and to test requests. The main endpoints are:

*   **`POST /predict/heart_disease`**: Predicts heart disease mortality.
*   **`POST /predict/diabetes`**: Predicts diabetes risk.
*   **`POST /predict/lung_cancer`**: Predicts lung cancer survival.
*   **`POST /predict/kidney_disease`**: Predicts chronic kidney disease.
*   **`POST /chat/query`**: Submits a question to the AI assistant chatbot.

### CORS (Cross-Origin Resource Sharing)

The API is configured to allow requests from any origin (`*`) during local development, so you should not encounter any CORS errors when calling it from a local frontend server (e.g., a React app running on `http://localhost:3000`).
