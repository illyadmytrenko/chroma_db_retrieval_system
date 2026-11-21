import bm25s
import numpy as np
import streamlit as st


# Caching bm25_retriever
@st.cache_resource
def get_bm25_retriever(corpus_tuple: tuple[str, ...]):
    corpus = list(corpus_tuple)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
    return retriever


def bm25_search(corpus: list[str], query: str) -> np.ndarray:
    if not corpus:
        return np.array([], dtype=np.float32)

    retriever = get_bm25_retriever(tuple(corpus))
    documents, scores = retriever.retrieve(
        bm25s.tokenize(query, show_progress=False),
        k=len(corpus),
        show_progress=False,
    )
    return documents[0], scores[0]


# Getting ranks from scores for RRF function
def scores_to_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    order = np.argsort(-scores if higher_is_better else scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# Getting ranks from order for RRF function
def order_to_ranks(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# RRF function
def get_rrf_top_n(
    dense_distances: list[float],
    sparse_orders: np.ndarray,
    n: int,
    k: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    dense_arr = np.asarray(dense_distances, dtype=np.float32)

    # Dense (ChromaDB): smaller distance = better match
    # Sparse (BM25): higher score = better match
    dense_ranks = scores_to_ranks(dense_arr, higher_is_better=False)
    sparse_ranks = order_to_ranks(sparse_orders)

    rrf_scores = (1.0 / (k + dense_ranks)) + (1.0 / (k + sparse_ranks))

    top_n_indices = np.argsort(-rrf_scores)[:n]
    top_n_scores = rrf_scores[top_n_indices]

    return top_n_indices, top_n_scores
