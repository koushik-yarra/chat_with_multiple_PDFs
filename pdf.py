import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize chat memory (stored in Streamlit session state)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function: Extract text from PDFs
def pdf_to_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function: Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function: Create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function: Create Gemini-powered QA chain with custom prompt
def conversation_chain():
    prompt_template = """
    You are an AI assistant who is expertise in document understanding and information retrieval. Make sure that you have to give the answers relevant to the question. do as much as you can  Use the context to answer the user's question as accurately as possible.
    If the answer is not found in the context, respond with: "Answer is not available in the context."

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

# Function: Handle user questions with memory + error handling
def get_answer_with_memory(user_question):
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("❗ Please upload and process PDFs first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    # Format chat history
    chat_history_text = ""
    for turn in st.session_state.chat_history:
        chat_history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    # Get answer from chain
    chain = conversation_chain()
    response = chain(
        {
            "input_documents": docs,
            "question": user_question,
            "chat_history": chat_history_text
        },
        return_only_outputs=True
    )

    # Save chat
    st.session_state.chat_history.append({
        "user": user_question,
        "assistant": response["output_text"]
    })

    # Display response
    st.write("💬 Assistant:", response["output_text"])

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.title("📄 Chat with Multiple PDFs")

    # Two tabs: Chat and History
    tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])

    with tab1:
        st.subheader("💬 Ask Questions About Your PDFs")

        user_question = st.text_input("Ask a question 👇", key="user_question")

        if user_question:
            get_answer_with_memory(user_question)

    with tab2:
        st.subheader("📜 Chat History")

        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(chat["user"])
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(chat["assistant"])

            # 🧹 Add Clear History button here
            if st.button("🗑️ Clear Chat History", key="clear_history"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
        else:
            st.info("No chat history yet. Ask questions in the Chat tab.")

    # Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.title("📎 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Submit & Process", key="submit_pdfs"):
            with st.spinner("Reading and indexing your PDFs..."):
                raw_text = pdf_to_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDFs processed and indexed!")

# Run the app
if __name__ == "__main__":
    main()
