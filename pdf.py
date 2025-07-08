import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langdetect import detect
import base64
import datetime

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "Auto"

# OCR function for scanned PDFs
def extract_text_with_ocr(pdf_file):
    text = ""
    images = convert_from_bytes(pdf_file.read())
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

# Extract text from PDFs (OCR fallback)
def pdf_to_text(pdf_docs):
    full_text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            if not text.strip():
                text = extract_text_with_ocr(pdf)
        except:
            text = extract_text_with_ocr(pdf)
        full_text += text
    return full_text

# Text splitting
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    return splitter.split_text(text)

# Vector DB
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# Prompt + Chain
def conversation_chain():
    prompt_template = """
    You are a helpful multilingual pdf assistant. Use the context below to answer the user's question in the same language or what ever the language they mention.
    If the answer is not found, say: "Answer is not available in the context."

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Answer retrieval with memory
def get_answer_with_memory(user_question):
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("Please upload and process PDFs first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(user_question)

    chat_history_text = "\n".join([
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in st.session_state.chat_history
    ])

    chain = conversation_chain()
    response = chain({
        "input_documents": docs,
        "question": user_question,
        "chat_history": chat_history_text
    }, return_only_outputs=True)

    st.session_state.chat_history.append({
        "user": user_question,
        "assistant": response["output_text"]
    })

    st.write("💬 Assistant:", response["output_text"])

# Download chat history
def download_chat_history():
    if not st.session_state.chat_history:
        st.warning("No chat history to download.")
        return
    filename = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    content = "\n\n".join([f"User: {c['user']}\nAssistant: {c['assistant']}" for c in st.session_state.chat_history])
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Chat History</a>'
    st.markdown(href, unsafe_allow_html=True)

# Streamlit App
def main():
    st.set_page_config(page_title="Multilingual PDF Chatbot", layout="wide")
    st.title("🌍 Multilingual Chat with PDFs")

    tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])

    with tab1:
        st.subheader("Ask a question in your language")
        st.selectbox("Select Input Language (optional):", ["Auto", "English", "Hindi", "Telugu", "Tamil", "Spanish", "German", "French", "Arabic", "Chinese"], key="language")
        user_question = st.text_input("Ask your question here 👇", key="user_question")
        if user_question:
            get_answer_with_memory(user_question)

    with tab2:
        st.subheader("📜 Chat History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(chat["user"])
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(chat["assistant"])
            if st.button("🗑️ Clear Chat History", key="clear_history"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
            download_chat_history()
        else:
            st.info("No chat history yet. Ask something in the Chat tab.")

    with st.sidebar:
        st.title("📎 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process", key="process"):
            with st.spinner("Processing your PDFs..."):
                raw_text = pdf_to_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDFs processed and indexed!")

if __name__ == "__main__":
    main()
