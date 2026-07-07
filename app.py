from pathlib import Path

import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.nlp_pipeline import (
    answer_question,
    compare_documents,
    extract_keywords,
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
        "Tip: upload one or more PDF/TXT papers. Multi-paper mode is supported."
    )


def read_uploaded_file(uploaded_file) -> str:
    """Read PDF or TXT content from a Streamlit uploaded file."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    return uploaded_file.read().decode("utf-8", errors="ignore")


def get_target_documents(target_name: str, documents: list[str], document_names: list[str]):
    """Return selected document(s) based on user choice."""
    if target_name == "All uploaded papers":
        return documents, document_names

    selected_index = document_names.index(target_name)
    return [documents[selected_index]], [document_names[selected_index]]


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
    st.subheader("Grounded summary")
    st.caption(
        "By default, summaries are generated separately for each uploaded paper."
    )

    summary_mode = st.radio(
        "Summary mode",
        ["Separate summaries", "Combined summary"],
        horizontal=True,
    )

    if summary_mode == "Separate summaries":
        for doc_name, doc_text in zip(document_names, documents):
            with st.expander(doc_name, expanded=len(documents) <= 2):
                summary = summarize_text(
                    doc_text,
                    max_sentences=max_summary_sentences,
                )
                st.markdown(summary)
    else:
        st.warning(
            "Combined summary mixes all uploaded papers. Use this only when you want an overview of the full document set."
        )
        summary = summarize_text(
            combined_text,
            max_sentences=max_summary_sentences,
        )
        st.markdown(summary)


with tabs[1]:
    st.subheader("Keywords and phrases")
    st.caption(
        "Keywords are shown separately by paper. You can also view combined keywords."
    )

    keyword_mode = st.radio(
        "Keyword mode",
        ["By paper", "Combined keywords"],
        horizontal=True,
    )

    if keyword_mode == "By paper":
        for doc_name, doc_text in zip(document_names, documents):
            with st.expander(doc_name, expanded=len(documents) <= 2):
                keywords = extract_keywords(doc_text, top_n=max_keywords)

                if not keywords.empty:
                    st.dataframe(
                        keywords,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning("No keywords found for this document.")
    else:
        keywords = extract_keywords(combined_text, top_n=max_keywords)

        if not keywords.empty:
            st.dataframe(
                keywords,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No keywords found. Try a longer document.")


with tabs[2]:
    st.subheader("Ask the paper")
    st.caption(
        "Ask a question in English or Chinese. You can target one paper or all uploaded papers."
    )

    target_options = ["All uploaded papers"] + document_names

    target_name = st.selectbox(
        "Question target",
        target_options,
        index=0,
    )

    example_questions = [
        "What material system is studied?",
        "What methods or characterization techniques are used?",
        "What are the main results?",
        "What limitations or future work are mentioned?",
        "What parameters or settings are used?",
        "这篇文章研究了什么材料体系？",
        "这篇文章用了什么方法或表征手段？",
        "这篇文章的主要结果是什么？",
        "这篇文章有什么局限或未来工作？",
        "这篇文章用了哪些参数或设定？",
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
            target_documents, target_document_names = get_target_documents(
                target_name,
                documents,
                document_names,
            )

            qa_result = answer_question(
                documents=target_documents,
                query=query,
                document_names=target_document_names,
                top_k=top_k,
                max_answer_sentences=4,
            )

            st.markdown(qa_result["answer"])

            evidence = qa_result.get("evidence", [])

            if evidence:
                with st.expander("Show retrieved evidence passages", expanded=False):
                    for i, item in enumerate(evidence, start=1):
                        source_index = item["document_index"]
                        source_name = target_document_names[source_index]
                        st.markdown(f"#### Evidence passage {i}")
                        st.caption(
                            f"Source: {source_name} | similarity: {item['score']:.3f}"
                        )
                        st.write(item["passage"])


with tabs[3]:
    st.subheader("Compare papers")
    st.caption(
        "Upload two or more papers to compare their research focus, methods, parameters, results, and limitations."
    )

    if len(documents) < 2:
        st.info(
            "Please upload at least two documents to use this comparison mode. "
            "You can also upload one paper and keep the sample text enabled for testing."
        )
    else:
        max_compare_snippets = st.slider(
            "Evidence snippets per dimension",
            min_value=1,
            max_value=5,
            value=3,
        )

        comparison = compare_documents(
            documents=documents,
            document_names=document_names,
            max_snippets_per_dimension=max_compare_snippets,
        )

        st.markdown("### Core comparison table")
        st.dataframe(
            comparison["table"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Keyword overlap and distinctive keywords")
        st.dataframe(
            comparison["keyword_table"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Detailed evidence by dimension")

        details = comparison["details"]

        for dimension, document_snippets in details.items():
            with st.expander(dimension, expanded=False):
                columns = st.columns(len(document_names))

                for column, doc_name in zip(columns, document_names):
                    with column:
                        st.markdown(f"**{doc_name}**")
                        snippets = document_snippets.get(doc_name, [])

                        if snippets:
                            for snippet in snippets:
                                st.markdown(f"- {snippet}")
                        else:
                            st.caption("Not clearly detected.")

        st.info(comparison["note"])


with tabs[4]:
    st.subheader("Materials-science domain hints")
    st.caption(
        "Domain hints are shown separately for each uploaded paper."
    )

    for doc_name, doc_text in zip(document_names, documents):
        with st.expander(doc_name, expanded=len(documents) == 1):
            hints = find_domain_hints(doc_text)

            for category, snippets in hints.items():
                st.markdown(f"### {category.replace('_', ' ').title()}")

                if snippets:
                    for snippet in snippets:
                        st.markdown(f"- {snippet}")
                else:
                    st.caption("No obvious hint detected.")


with tabs[5]:
    st.subheader("Document preview")

    for name, doc in zip(document_names, documents):
        with st.expander(name, expanded=False):
            st.write(doc[:5000] + ("..." if len(doc) > 5000 else ""))
