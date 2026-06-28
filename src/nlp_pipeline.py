from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_utils import clean_text, split_passages, split_sentences


def summarize_text(text: str, max_sentences: int = 5) -> str:
    """Create an extractive summary using sentence-level TF-IDF scores."""
    sentences = split_sentences(text)

    if not sentences:
        return "No readable sentence was found."

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(sentences)
        scores = matrix.sum(axis=1).A1
    except ValueError:
        return " ".join(sentences[:max_sentences])

    top_indices = scores.argsort()[::-1][:max_sentences]
    top_indices = sorted(top_indices)

    return " ".join(sentences[i] for i in top_indices)


def extract_keywords(text: str, top_n: int = 15) -> pd.DataFrame:
    """Extract top TF-IDF keywords and short phrases from one document."""
    text = clean_text(text)

    if not text:
        return pd.DataFrame(columns=["keyword", "score"])

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500,
        )
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        return pd.DataFrame(columns=["keyword", "score"])

    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]

    ranked = sorted(
        zip(feature_names, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    return pd.DataFrame(ranked, columns=["keyword", "score"])


def retrieve_passages(
    documents: List[str],
    query: str,
    top_k: int = 4,
) -> List[Dict[str, object]]:
    """Retrieve passages that are most similar to the query."""
    all_passages = []
    passage_to_doc = []

    for doc_index, doc in enumerate(documents):
        for passage in split_passages(doc):
            all_passages.append(passage)
            passage_to_doc.append(doc_index)

    if not all_passages or not query.strip():
        return []

    corpus = all_passages + [query]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return []

    query_vector = matrix[-1]
    passage_matrix = matrix[:-1]
    similarities = cosine_similarity(query_vector, passage_matrix).ravel()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for index in top_indices:
        results.append(
            {
                "document_index": passage_to_doc[index],
                "passage": all_passages[index],
                "score": float(similarities[index]),
            }
        )

    return results


DOMAIN_KEYWORDS = {
    "materials_system": [
        "film",
        "thin film",
        "heterostructure",
        "ferroelectric",
        "semiconductor",
        "skyrmion",
        "vdw",
        "van der waals",
        "oxide",
        "chalcogenide",
        "silicon",
        "wafer",
    ],
    "fabrication_or_synthesis": [
        "synthesis",
        "growth",
        "deposition",
        "annealing",
        "lithography",
        "etching",
        "sputtering",
        "cvd",
        "mocvd",
        "ald",
        "exfoliation",
    ],
    "characterization": [
        "sem",
        "tem",
        "fib",
        "xrd",
        "raman",
        "afm",
        "ebsd",
        "shg",
        "ellipsometry",
        "metrology",
        "microscopy",
    ],
    "results_or_performance": [
        "increase",
        "decrease",
        "improvement",
        "mobility",
        "coercive",
        "polarization",
        "switching",
        "stability",
        "efficiency",
        "accuracy",
        "defect density",
    ],
    "limitations_or_future_work": [
        "limitation",
        "challenge",
        "future work",
        "however",
        "requires",
        "uncertain",
        "noise",
        "artifact",
        "scalability",
    ],
}


def find_domain_hints(text: str, max_snippets_per_category: int = 4) -> Dict[str, List[str]]:
    """Find short snippets related to common materials-science paper sections."""
    sentences = split_sentences(text)
    hints: Dict[str, List[str]] = {}

    for category, keywords in DOMAIN_KEYWORDS.items():
        category_hits = []
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(keyword in lower_sentence for keyword in keywords):
                category_hits.append(sentence)
            if len(category_hits) >= max_snippets_per_category:
                break
        hints[category] = category_hits

    return hints
