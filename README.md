# 📄 ChatPDF – Chat with Your PDFs

ChatPDF is a **Streamlit** application that allows you to upload PDF files, process their content into vector embeddings using **Google Generative AI** embeddings, and interact with them in a conversational manner. You can ask natural language questions, and the app will retrieve relevant chunks from your PDF and generate AI-powered responses.

## 🚀 Features
- 📂 **Multiple PDF Uploads** – Upload one or more PDFs at once.  
- ✂ **Smart Text Chunking** – Splits PDF text into manageable chunks for efficient retrieval.  
- 🧠 **Semantic Search** – Uses FAISS vector store for fast and relevant content retrieval.  
- 💬 **Conversational Memory** – Maintains context across questions in a session.  
- ⚡ **Google Generative AI** – Powered by `gemini-1.5-flash` for fast, accurate answers.  
- 🎨 **Custom Chat UI** – Styled user and bot messages for a chat-like experience.  

## 🛠 Tech Stack
- **Frontend & UI:** [Streamlit](https://streamlit.io/)  
- **PDF Processing:** [PyPDF2](https://pypi.org/project/PyPDF2/)  
- **LLM & Embeddings:** [LangChain](https://www.langchain.com/), [Google Generative AI](https://ai.google/)  
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)  
- **Environment Management:** `python-dotenv`

## 📦 Installation
```bash
git clone https://github.com/your-username/chatpdf.git
cd chatpdf
```
```bash
# Windows
python -m venv venv
venv\Scripts\activate
# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```
💡 Get your API key from [Google AI Studio](https://makersuite.google.com/).

## ▶️ Run the Application
```bash
streamlit run app.py
```

## 📂 Project Structure
```
├── app.py                # Main Streamlit application
├── template.py           # UI templates and CSS for chat messages
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not committed to Git)
└── README.md             # Project documentation
```

## 🖥 Usage
1. Upload PDFs using the sidebar file uploader.  
2. Click **"Proceed"** to process and store embeddings.  
3. Type your question in the input box at the top.  
4. Receive an AI-generated response, with context maintained across turns.  

## ⚠️ Notes
- Your **Google API key** must have access to the Generative AI API.  
- Large PDF files may take longer to process.  
- FAISS stores embeddings in memory only; add persistence if needed.  


