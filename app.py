import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template import css, bot_template, user_template




GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.warning("Please upload PDFs and click Proceed before asking questions.")
        return

    with st.spinner("Thinking..."):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

    # st.session_state.query = ""  

def display_chat_history():
    """Always show chat history if available."""
    if st.session_state.chat_history:
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            original_index = len(st.session_state.chat_history) - 1 - i
            if original_index % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

st.write(css, unsafe_allow_html=True)


if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "query" not in st.session_state:
    st.session_state.query = ""

st.header("ChatPDF")
query = st.text_input("Ask a question ...", value=st.session_state.query, key="query_input")

# Handle query only if non-empty
if query.strip():
    handle_userinput(query)


display_chat_history()

with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDF(s) here", accept_multiple_files=True)



    if st.button("Proceed"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF before proceeding.")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

            st.info("Processing complete! You can now ask a question.")
            