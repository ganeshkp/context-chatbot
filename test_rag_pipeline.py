from langchain_core.documents import Document

from context_chatbot import mmr_select, compress_context, build_multistage_retrieval


class DummyDoc:
    def __init__(self, content):
        self.page_content = content


class DummyVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k=5):
        results = []
        q = query.lower().split()
        for doc in self.docs:
            score = sum(1 for token in q if token in doc.page_content.lower().split())
            results.append((doc, float(score)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]


def test_mmr_select_keeps_diverse_documents():
    docs = [
        DummyDoc("alpha beta gamma delta"),
        DummyDoc("alpha beta epsilon zeta"),
        DummyDoc("omega theta iota kappa"),
    ]

    selected = mmr_select("alpha beta", docs, final_k=2)

    assert len(selected) == 2
    assert all(doc.page_content in {d.page_content for d in docs} for doc in selected)
    assert selected[0].page_content != selected[1].page_content


def test_compress_context_limits_output_length():
    docs = [
        DummyDoc("word " * 200),
        DummyDoc("second chunk with some useful facts " * 50),
    ]

    compressed = compress_context(docs, max_tokens=60)

    assert len(compressed.split()) <= 60 * 2


def test_build_multistage_retrieval_returns_ranked_docs():
    docs = [
        Document(
            page_content="The project uses Python and machine learning for document analysis."
        ),
        Document(
            page_content="The project uses Python to build a chatbot for reports."
        ),
        Document(
            page_content="The company ships marketing content for a retail campaign."
        ),
    ]

    results = build_multistage_retrieval(
        "python chatbot for document analysis",
        docs,
        DummyVectorStore(docs),
        final_k=2,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)
    assert any("chatbot" in doc.page_content.lower() for doc in results)
