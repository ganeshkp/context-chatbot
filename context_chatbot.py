import streamlit as st
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = st.secrets["api_keys"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Relevance Scoring"
groq_api_key = st.secrets["api_keys"]["GROQ_API_KEY"]

# Clean text
def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned_lines)

# Custom retriever with relevance scoring
def retrieve_with_scores(query, vectorstore, k=3, score_threshold=0.7):
    """Retrieve documents with relevance scores and filter by threshold"""
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    # Filter documents by score threshold
    filtered_docs = [doc for doc, score in docs_and_scores if score >= score_threshold]
    
    # Display retrieval info in sidebar
    st.sidebar.subheader("Retrieval Information")
    st.sidebar.write(f"Found {len(docs_and_scores)} documents, {len(filtered_docs)} meet threshold")
    for i, (doc, score) in enumerate(docs_and_scores):
        st.sidebar.write(f"Document {i+1} - Score: {score:.3f}")
        st.sidebar.text(doc.page_content[:100] + "...")
    
    return filtered_docs

st.title("Context Chat Bot")

uploaded_file = st.file_uploader("Choose a text file", type="txt", accept_multiple_files=False)
docs = []

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_file.txt", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = TextLoader("temp_file.txt")
    docs = loader.load()
    
    # Apply cleanup to each document
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    
    # Embedding the documents    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store embeddings in vector DB
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Create a retriever with scoring
    retriever = RunnableLambda(
        lambda query: retrieve_with_scores(query, vectorstore, k=3, score_threshold=0.7)
    ).bind()
    
    # Create a llm for the chat
    llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")
    
    message = """
    Answer the questions based only on the provided context.
    If the question cannot be answered with the context, simply say "I don't have that information."
    Do not mention the context in your response.
    
    Question: {question}
    
    Context: {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    # Format documents for context
    def format_docs(docs):
        if not docs:
            return "No relevant information found"
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | llm
    
    # Add a query input and response section
    st.subheader("Ask a question about the document")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        response = rag_chain.invoke(user_question)
        st.subheader("Answer:")
        st.write(response.content)
        
else:
    st.write("Please upload a text document to begin")