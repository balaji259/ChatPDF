import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template import css, bot_template, user_template

load_dotenv()

def get_pdf_text(pdf_docs):
    text=""
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

    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore=FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

st.write(css, unsafe_allow_html=True)

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

st.header("ChatPDF")
query = st.text_input("Ask a question ... (Upload the pdf(s) and click proceed before asking a question)")

if query:
    handle_userinput(query)

with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDF's here and click proceed",accept_multiple_files=True)
    if st.button("Proceed"):
        with st.spinner("Processing"):
            
            raw_text = get_pdf_text(pdf_docs)
            
            text_chunks= get_text_chunks(raw_text)
            
            vectorstore = get_vectorstore(text_chunks) 

            st.session_state.conversation = get_conversation_chain(vectorstore)