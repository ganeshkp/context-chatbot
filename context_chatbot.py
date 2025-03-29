import streamlit as st
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = st.secrets["api_keys"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A With Relevance Scoring"
groq_api_key = st.secrets["api_keys"]["GRAQ_API_KEY"]

# Clean text
def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned_lines)

# Custom retriever with relevance threshold
def retrieve_with_scores(query: str, vectorstore, k: int = 3, score_threshold: float = 0.7) -> List[Document]:
    """Retrieve documents with relevance scores and apply threshold filtering"""
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    filtered_docs = [doc for doc, score in docs_and_scores if score >= score_threshold]
    
    # Logging for debugging
    st.sidebar.subheader("Retrieval Debug Info")
    st.sidebar.write(f"Retrieved {len(docs_and_scores)} documents, {len(filtered_docs)} passed threshold")
    for i, (doc, score) in enumerate(docs_and_scores):
        st.sidebar.write(f"Doc {i+1} - Score: {score:.3f}")
        st.sidebar.text(doc.page_content[:200] + "...")
    
    return filtered_docs

uploaded_file = st.file_uploader("Choose a text file", type="txt", accept_multiple_files=False)

if uploaded_file is not None:
    # Save and load the uploaded file
    with open("temp_file.txt", "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = TextLoader("temp_file.txt")
    docs = loader.load()
    
    # Clean and split documents
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    
    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Create LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")
    
    # Configure retrieval with scoring
    retriever = RunnableLambda(
        lambda query: retrieve_with_scores(query, vectorstore, k=3, score_threshold=0.7)
    )
    
    # Enhanced prompt with relevance awareness
    message = """
    Answer the question based only on the provided context. 
    The context comes with relevance scores (higher is better).
    If no context is provided or scores are low, say "I don't have enough relevant information."
    
    Question: {question}
    
    Context: {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])
    
    # Chain with formatted context including scores
    def format_docs(docs):
        return "\n\n".join([
            f"Document {i+1} (Relevance: {score:.3f}):\n{doc.page_content}"
            for i, (doc, score) in enumerate(docs)
        ])
    
    rag_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | llm
    
    # Query interface
    st.subheader("Ask a question about the document")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        response = rag_chain.invoke(user_question)
        st.subheader("Answer:")
        st.write(response.content)
else:
    st.write("Please upload a text document to begin")