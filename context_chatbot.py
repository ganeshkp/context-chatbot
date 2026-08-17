import streamlit as st
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

load_dotenv()


def get_config_value(key, section="api_keys"):
    """Read config from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(section, {}).get(key)
    except Exception:
        value = None

    return value or os.getenv(key)


def set_env_from_config(key):
    value = get_config_value(key)
    if value:
        os.environ[key] = value
    return value


## Langsmith Tracking
langchain_api_key = set_env_from_config("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true" if langchain_api_key else "false"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Relevance Scoring"
openai_api_key = set_env_from_config("OPENAI_API_KEY")
openai_chat_model = get_config_value("OPENAI_CHAT_MODEL") or "gpt-5.6-luna"
openai_embedding_model = get_config_value("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"


# Clean text
def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned_lines)


# Custom retriever that shows scores but doesn't filter
def retrieve_with_scores(query, vectorstore, k=3):
    """Retrieve documents with relevance scores (no filtering)"""
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    # Display retrieval info in sidebar
    st.sidebar.subheader("Retrieval Information")
    st.sidebar.write(f"Retrieved {len(docs_and_scores)} documents:")
    for i, (doc, score) in enumerate(docs_and_scores):
        st.sidebar.write(f"Document {i+1} - Score: {score:.3f}")
        st.sidebar.text(doc.page_content[:100] + "...")

    # Return just the documents without filtering
    return [doc for doc, score in docs_and_scores]


st.title("Context Chat Bot")

if not openai_api_key:
    st.error(
        "Missing OPENAI_API_KEY. Add it to Streamlit secrets under api_keys "
        "or set it as an environment variable."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a text file", type="txt", accept_multiple_files=False
)
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
    embeddings = OpenAIEmbeddings(model=openai_embedding_model)

    # Store embeddings in vector DB
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Create a retriever with scoring (no threshold filter)
    retriever = RunnableLambda(
        lambda query: retrieve_with_scores(query, vectorstore, k=3)
    ).bind()

    # Create a llm for the chat
    llm = ChatOpenAI(model=openai_chat_model)

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
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # Add a query input and response section
    st.subheader("Ask a question about the document")
    user_question = st.text_input("Enter your question:")

    if user_question:
        response = rag_chain.invoke(user_question)
        st.subheader("Answer:")
        st.write(response.content)

else:
    st.write("Please upload a text document to begin")
