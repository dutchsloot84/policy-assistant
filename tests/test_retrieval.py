from src.core.retrieval import Retriever
from src.store.faiss_store import FaissVectorStore, Metadata


def test_retrieval_orders_by_score(tmp_path):
    store = FaissVectorStore(index_path=tmp_path / "index.faiss", meta_path=tmp_path / "meta.pkl")
    # Pre-populate index manually by adding normalized vectors.
    store.add(
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
        metadatas=[
            Metadata(document_id="doc1", chunk_id="c1", text="policy alpha", source="doc1.pdf"),
            Metadata(document_id="doc2", chunk_id="c2", text="policy beta", source="doc2.pdf"),
        ],
    )
    retriever = Retriever(store)
    results = retriever.search([1.0, 0.0], top_k=2, redact=False)
    assert results[0].source == "doc1.pdf"
    assert len(results) == 2
