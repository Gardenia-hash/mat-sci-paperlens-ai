from pathlib import Path

import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.nlp_pipeline import (
    extract_keywords,
    retrieve_passages,
    summarize_text,
    find_domain_hints,
)
from src.text_utils import clean_text


st.set_page_config(
    page_title="MatSci PaperLens AI",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 MatSci PaperLens AI")
st.caption("A lightweight AI/NLP assistant for materials-science and semiconductor papers.")

with st.sidebar:
    st.header("Settings")
    max_summary_sentences = st.slider("Summary sentences", 3, 10, 5)
    max_keywords = st.slider("Number of keywords", 5, 30, 15)
    top_k = st.slider("Retrieved passages", 1, 8, 4)

    st.markdown("---")
    st.markdown(
        "Tip: upload a PDF paper, or use the built-in sample text to test the app."
    )


def read_uploaded_file(uploaded_file) -> str:
    """Read PDF or TXT content from a Streamlit uploaded file."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    return uploaded_file.read().decode("utf-8", errors="ignore")


uploaded_files = st.file_uploader(
    "Upload one or more PDF/TXT papers",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

use_sample = st.checkbox("Use sample materials-science text", value=not uploaded_files)

documents = []
document_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        raw_text = read_uploaded_file(uploaded_file)
        text = clean_text(raw_text)
        if text:
            documents.append(text)
            document_names.append(uploaded_file.name)

if use_sample:
    sample_path = Path("data/sample_materials_text.txt")
    if sample_path.exists():
        sample_text = clean_text(sample_path.read_text(encoding="utf-8"))
        documents.append(sample_text)
        document_names.append("sample_materials_text.txt")

if not documents:
    st.info("Upload a paper or enable the sample text to begin.")
    st.stop()

combined_text = "\n\n".join(documents)

tabs = st.tabs(["Summary", "Keywords", "Ask / Search", "Domain Hints", "Text Preview"])

with tabs[0]:
    st.subheader("Extractive summary")
    summary = summarize_text(combined_text, max_sentences=max_summary_sentences)
    st.markdown(summary)

with tabs[1]:
    st.subheader("Top keywords and phrases")
    keywords = extract_keywords(combined_text, top_n=max_keywords)
    if not keywords.empty:
        st.dataframe(keywords, use_container_width=True)
    else:
        st.warning("No keywords found. Try a longer document.")

with tabs[2]:
    st.subheader("Question-style passage retrieval")
    query = st.text_input(
        "Ask a question about the uploaded paper(s)",
        value="What material system and characterization methods are discussed?",
    )

    if query:
        results = retrieve_passages(documents, query=query, top_k=top_k)
        for i, item in enumerate(results, start=1):
            st.markdown(f"### Result {i}")
            source_index = item["document_index"]
            source_name = document_names[source_index]
            st.caption(f"Source: {source_name} | similarity: {item['score']:.3f}")
            st.write(item["passage"])

with tabs[3]:
    st.subheader("Materials-science domain hints")
    hints = find_domain_hints(combined_text)
    for category, snippets in hints.items():
        with st.expander(category.replace("_", " ").title(), expanded=True):
            if snippets:
                for snippet in snippets:
                    st.write("- " + snippet)
            else:
                st.caption("No obvious hint detected.")

with tabs[4]:
    st.subheader("Document preview")
    for name, doc in zip(document_names, documents):
        with st.expander(name, expanded=False):
            st.write(doc[:5000] + ("..." if len(doc) > 5000 else ""))
