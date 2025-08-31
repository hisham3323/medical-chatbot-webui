# Medical AI Chatbot

A web-based medical AI chatbot that provides diagnostic suggestions and prescription information, leveraging a RAG (Retrieval-Augmented Generation) architecture. This project features a Python FastAPI backend, a vanilla JavaScript frontend, and Supabase for user authentication and data persistence.

---
## ✨ Features

- **Conversational AI:** Utilizes OpenRouter with the Dolphin 3.0 Mistral model for intelligent medical dialogue.
- **User Authentication:** Secure user signup and login system handled by Supabase Auth.
- **Persistent Patient Profiles:** User medical data (age, conditions, etc.) is collected at signup and stored in a Supabase PostgreSQL database.
- **RAG Architecture:** Integrates a Pinecone vector database as a knowledge base to provide contextually relevant information to the LLM.
- **Web-Based UI:** A clean and simple chat interface built with HTML, CSS, and JavaScript.

---
## 🛠️ Tech Stack

### Backend
- **Python 3.11+**
- **FastAPI:** For the web server framework.
- **Uvicorn:** As the ASGI server.
- **Supabase:** For the PostgreSQL database and user authentication.
- **Pinecone:** For the vector database / knowledge base.
- **OpenRouter:** As the LLM inference provider.
- **LangChain:** For orchestrating the RAG components.
- **Pydantic:** For data validation.

### Frontend
- **HTML5**
- **CSS3**
- **Vanilla JavaScript**
- **Supabase-js:** For client-side authentication.

---
## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hisham3323/medical-chatbot-webui.git](https://github.com/hisham3323/medical-chatbot-webui.git)
    cd medical-chatbot-webui
    ```

2.  **Set up the backend:**
    - Create and activate a Python virtual environment:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      ```
    - Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
    - Create a `.env` file inside the `backend` folder and add your API keys:
      ```env
      PINECONE_API_KEY="YOUR_KEY_HERE"
      SUPABASE_URL="YOUR_URL_HERE"
      SUPABASE_KEY="YOUR_SERVICE_ROLE_KEY_HERE"
      OPENROUTER_API_KEY="YOUR_KEY_HERE"
      ```

3.  **Set up the frontend:**
    - Open the `frontend/index.html` file.
    - Replace the placeholder values for `supabaseUrl` and `supabaseAnonKey` with your credentials.

4.  **Run the server:**
    ```bash
    cd backend
    uvicorn chat_server:app --reload --host 0.0.0.0 --port 5000
    ```

5.  Open `frontend/index.html` in your web browser to use the application.