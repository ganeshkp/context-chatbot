import math
import os
import re
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


# Langsmith tracking configuration.
langchain_api_key = set_env_from_config("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true" if langchain_api_key else "false"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Multi-Stage Retrieval"
openai_api_key = set_env_from_config("OPENAI_API_KEY")
openai_chat_model = get_config_value("OPENAI_CHAT_MODEL") or "gpt-5.6-luna"
openai_embedding_model = (
    get_config_value("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
)


def clean_text(text):
    """Normalize whitespace and strip empty lines."""
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned_lines)


def _tokenize(text):
    return re.findall(r"\b\w+\b", (text or "").lower())


def _doc_text(doc):
    return getattr(doc, "page_content", "") or ""


def _doc_overlap_score(left, right):
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens and not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def bm25_rank(query, documents):
    """Lightweight BM25-style scoring without adding a heavy runtime dependency."""
    docs = [doc for doc in documents if _doc_text(doc)]
    if not docs:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return [(doc, 0.0) for doc in docs]

    doc_tokens = [_tokenize(_doc_text(doc)) for doc in docs]
    num_docs = len(doc_tokens)
    avgdl = sum(len(tokens) for tokens in doc_tokens) / max(1, num_docs)

    doc_freq = {}
    for tokens in doc_tokens:
        seen = set()
        for term in set(tokens):
            if term not in seen:
                doc_freq[term] = doc_freq.get(term, 0) + 1
                seen.add(term)

    idf = {
        term: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        for term, freq in doc_freq.items()
    }

    scored = []
    k1 = 1.5
    b = 0.75
    for doc, tokens in zip(docs, doc_tokens):
        counts = Counter(tokens)
        score = 0.0
        for term in set(query_tokens):
            if term not in idf or term not in counts:
                continue
            tf = counts[term]
            denom = tf + k1 * (1 - b + b * len(tokens) / max(1, avgdl))
            score += idf[term] * ((tf * (k1 + 1)) / denom)
        scored.append((doc, float(score)))

    return sorted(scored, key=lambda item: item[1], reverse=True)


def dense_rank(query, vectorstore, k=12):
    """Dense retrieval stage using the existing FAISS index."""
    if vectorstore is None or not hasattr(vectorstore, "similarity_search_with_score"):
        return []
    try:
        return vectorstore.similarity_search_with_score(query, k=k)
    except Exception:
        return []


def _doc_identity(doc):
    return id(doc)


def hybrid_search(query, vectorstore, documents, candidate_k=12):
    """Stage 1: hybrid BM25 + dense retrieval to maximize recall."""
    dense_results = dense_rank(query, vectorstore, k=candidate_k)
    bm25_results = bm25_rank(query, documents)

    dense_scores = {_doc_identity(doc): float(score) for doc, score in dense_results}
    bm25_scores = {_doc_identity(doc): float(score) for doc, score in bm25_results}

    combined = {}
    for doc in documents:
        key = _doc_identity(doc)
        dense_score = dense_scores.get(key, 0.0)
        bm25_score = bm25_scores.get(key, 0.0)
        combined[key] = dense_score + bm25_score

    ranked_keys = sorted(combined, key=lambda key: combined[key], reverse=True)
    ordered_docs = []
    for key in ranked_keys[:candidate_k]:
        for doc in documents:
            if _doc_identity(doc) == key:
                ordered_docs.append(doc)
                break
    return ordered_docs


def cross_encoder_rerank(query, documents):
    """Stage 2: re-rank candidates by a cross-encoder-style relevance signal.

    A full cross-encoder model (for example, a sentence-transformer reranker)
    can be dropped in here later. This fallback keeps the app working in the
    current environment while preserving the production pattern.
    """
    if not documents:
        return []

    q_tokens = set(_tokenize(query))
    scored = []
    for doc in documents:
        text = _doc_text(doc)
        answer_tokens = set(_tokenize(text))
        overlap = len(q_tokens & answer_tokens)
        lexical_score = overlap / max(1, len(q_tokens | answer_tokens))
        score = lexical_score
        if hasattr(doc, "metadata") and "retrieval_score" in doc.metadata:
            score += float(doc.metadata["retrieval_score"])
        scored.append((doc, float(score)))

    return sorted(scored, key=lambda item: item[1], reverse=True)


def mmr_select(query, documents, final_k=3, lambda_param=0.5):
    """Stage 3: MMR reduces redundancy and keeps the final context diverse."""
    if not documents:
        return []

    selected = []
    remaining = list(documents)
    q_tokens = set(_tokenize(query))

    if not remaining:
        return selected

    # Seed with the document with the strongest query overlap.
    def query_coverage(doc):
        text = _doc_text(doc)
        return len(set(_tokenize(text)) & q_tokens) / max(1, len(q_tokens))

    first = max(remaining, key=query_coverage)
    selected.append(first)
    remaining = [doc for doc in remaining if doc is not first]

    while len(selected) < min(final_k, len(documents)):
        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:
            candidate_text = _doc_text(candidate)
            q_score = len(set(_tokenize(candidate_text)) & q_tokens) / max(
                1, len(q_tokens)
            )
            max_similarity = 0.0
            for chosen in selected:
                max_similarity = max(
                    max_similarity,
                    _doc_overlap_score(_doc_text(chosen), candidate_text),
                )
            mmr_score = (lambda_param * q_score) - (
                (1.0 - lambda_param) * max_similarity
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining = [doc for doc in remaining if doc is not best_candidate]

    return selected


def compress_context(documents, max_tokens=400):
    """Stage 4: trim or summarize retrieved chunks to fit the model context budget."""
    if not documents:
        return ""

    chunks = []
    total_tokens = 0
    for doc in documents:
        tokens = _tokenize(_doc_text(doc))
        remaining = max_tokens - total_tokens
        if remaining <= 0:
            break
        # Keep the most relevant leading section, then trim aggressively.
        chunk_tokens = tokens[:remaining]
        chunks.append(" ".join(chunk_tokens))
        total_tokens += len(chunk_tokens)

    return "\n\n".join(chunks).strip()


def build_multistage_retrieval(
    query, documents, vectorstore, final_k=3, candidate_k=12
):
    """Combine the four production retrieval stages into a single pipeline."""
    if not documents:
        return []

    stage_1 = hybrid_search(query, vectorstore, documents, candidate_k=candidate_k)
    stage_2 = cross_encoder_rerank(query, stage_1)
    stage_3 = mmr_select(query, [doc for doc, _ in stage_2], final_k=final_k)
    return stage_3


def retrieve_with_scores(query, vectorstore, documents, k=3):
    """Return the final ranked documents for the app to pass into the LLM."""
    ranked = build_multistage_retrieval(query, documents, vectorstore, final_k=k)
    return ranked


def build_rag_chain(documents, vectorstore, llm):
    """Assemble the retrieval + generation pipeline for the Streamlit app."""

    def retrieve_and_format(query):
        docs = retrieve_with_scores(query, vectorstore, documents, k=3)
        st.sidebar.subheader("Retrieval Information")
        st.sidebar.write(f"Retrieved {len(docs)} documents for the final prompt:")
        for i, doc in enumerate(docs):
            st.sidebar.write(f"Document {i + 1}")
            st.sidebar.text(
                _doc_text(doc)[:200] + ("..." if len(_doc_text(doc)) > 200 else "")
            )
        return compress_context(docs, max_tokens=700)

    message = """
    Answer the questions based only on the provided context.
    If the question cannot be answered with the context, simply say "I don't have that information."
    Do not mention the context in your response.

    Question: {question}

    Context: {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain = (
        {
            "context": RunnableLambda(lambda q: retrieve_and_format(q)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    return rag_chain


def main():
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

    if uploaded_file is None:
        st.write("Please upload a text document to begin")
        return

    with open("temp_file.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = TextLoader("temp_file.txt")
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=openai_embedding_model)
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    llm = ChatOpenAI(model=openai_chat_model)
    rag_chain = build_rag_chain(document_chunks, vectorstore, llm)

    st.subheader("Ask a question about the document")
    user_question = st.text_input("Enter your question:")

    if user_question:
        response = rag_chain.invoke(user_question)
        st.subheader("Answer:")
        st.write(response.content)


if __name__ == "__main__":
    main()
