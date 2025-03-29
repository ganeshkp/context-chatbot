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
os.environ["LANGCHAIN_API_KEY"]=st.secrets["api_keys"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Huggingface"
groq_api_key=st.secrets["api_keys"]["GROQ_API_KEY"]

# Clean text
def clean_text(text):
    lines = text.splitlines()  # Split into lines
    cleaned_lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines and spaces
    cleaned_text = " ".join(cleaned_lines)  # Join into a single string with spaces
    return cleaned_text

st.title("Context Chat Bot")


uploaded_file=st.file_uploader("Choose a text file", type="txt", accept_multiple_files=False)
docs=[]

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_file.txt", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader=TextLoader("temp_file.txt")
    docs=loader.load()
    
    # Apply cleanup to each document
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        
    # Split the documents
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents=text_splitter.split_documents(docs)
    
    #Embedding the documents    
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    #Store embeddings in vector DB
    vectorstore=FAISS.from_documents(documents, embeddings)

    #Create a retriever    
    retriever=RunnableLambda(vectorstore.similarity_search).bind(k=1)
    
    # Create a llm for the chat
    llm=ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")
    
    message = """
        Answer the questions based on the provided context only.
        dont answer if questions are out of context and say i dont have answer.
        But dont mention about context
        Please provide the most accurate response based on the question
        

        {question}

        Context:
        {context}
        """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain={"context":retriever,"question":RunnablePassthrough()}|prompt|llm
    
    # Add a query input and response section
    st.subheader("Ask a question about the document")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        response = rag_chain.invoke(user_question)
        st.subheader("Answer:")
        st.write(response.content)
        
else:
    st.write("Please upload text document for the context")
