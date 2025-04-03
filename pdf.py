import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load the environment variables from .env file
load_dotenv()

# set the api key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

 
def pdf_to_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model='gemini-2.0-flash',temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def get_answer(question):
    # Load stored FAISS index with safe deserialization
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    
    # Perform similarity search
    docs = vector_store.similarity_search(question)

    # Get response from conversation chain
    chain = conversation_chain()
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    st.write('Reply:', response['output_text'])



def main():
    st.set_page_config(page_title="CHAT WITH MULTIPLE PDFs")
    st.title("CHAT WITH PDFs")
    user_question=st.text_input("Enter your question here",key="user_question")
    if user_question:
        get_answer(user_question)

    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs=st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("submit & process"):
            with st.spinner("Processing..."):
                raw_text=pdf_to_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")


if __name__=="__main__":
    main()