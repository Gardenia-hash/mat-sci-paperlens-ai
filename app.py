from pathlib import Path

import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.nlp_pipeline import (
    answer_question,
    compare_documents,
    extract_keywords,
    extract_sections,
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

tabs = st.tabs(
    [
        "Summary",
        "Keywords",
        "Ask / Search",
        "Compare Papers",
        "Domain Hints",
        "Text Preview",
    ]
)

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
    st.subheader("Ask the paper")
    st.caption(
        "Ask a question in English or Chinese. The answer is generated from retrieved paper evidence."
    )

    example_questions = [
        "What material system is studied?",
        "What methods or characterization techniques are used?",
        "What are the main results?",
        "What limitations or future work are mentioned?",
        "这篇文章研究了什么材料体系？",
        "这篇文章用了什么方法或表征手段？",
        "这篇文章的主要结果是什么？",
        "这篇文章有什么局限或未来工作？",
    ]

    selected_example = st.selectbox(
        "Example questions",
        example_questions,
        index=0,
    )

    with st.form("qa_form"):
        query = st.text_area(
            "Your question",
            value=selected_example,
            height=100,
        )

        submitted = st.form_submit_button("Generate grounded answer")

    if submitted:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            qa_result = answer_question(
                documents=documents,
                query=query,
                document_names=document_names,
                top_k=top_k,
                max_answer_sentences=4,
            )

            st.markdown(qa_result["answer"])

            evidence = qa_result.get("evidence", [])

            if evidence:
                with st.expander("Show retrieved evidence passages", expanded=False):
                    for i, item in enumerate(evidence, start=1):
                        source_index = item["document_index"]
                        source_name = document_names[source_index]
                        st.markdown(f"#### Evidence passage {i}")
                        st.caption(
                            f"Source: {source_name} | similarity: {item['score']:.3f}"
                        )
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
