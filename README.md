# ❤️ Precision Medicine: Heart Disease Prediction App

## About The Project

This project is an end-to-end data science application that predicts the likelihood of mortality from heart failure based on a patient's clinical records. 

The application features a machine learning model that has been tuned for high performance and includes model explainability using SHAP (SHapley Additive exPlanations). This allows users to not only get a prediction but also understand the key factors driving that prediction, aligning with the principles of precision medicine.

**Live Demo:** [Link to your deployed app will go here]

### Key Features:
*   **High-Performance Model:** A tuned Random Forest Classifier achieving an AUC of **0.928** on the test set.
*   **Interactive UI:** A user-friendly web interface built with Streamlit.
*   **Explainable AI (XAI):** Integrated SHAP waterfall plots to explain each prediction, making the model's decisions transparent.
*   **End-to-End Workflow:** Demonstrates the full data science lifecycle from data analysis and preprocessing to model training, hyperparameter tuning, and deployment.

### Tech Stack
*   **Python**
*   **Pandas & NumPy** for data manipulation
*   **Scikit-learn** for modeling and preprocessing
*   **XGBoost** & **Random Forest** for model training
*   **SHAP** for model explainability
*   **Streamlit** for the web application interface
*   **Docker** for containerization

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.9+
*   Git

### Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/satviksrivatsav/precision-medicine.git
    cd precision-medicine
    ```

2.  **Create and activate a virtual environment:**
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

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Navigate to the backend directory:**
    ```sh
    cd backend
    ```

2.  **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

Your web browser should automatically open with the application running!