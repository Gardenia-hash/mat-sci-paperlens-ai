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


UI_TEXT = {
    "English": {
        "app_title": "MatSci PaperLens AI",
        "app_subtitle": "A grounded AI/NLP assistant for materials-science and semiconductor papers.",
        "hero_badge": "Local · Evidence-based · Multi-paper",
        "upload_label": "Upload one or more PDF/TXT papers",
        "sample_checkbox": "Use sample materials-science text",
        "settings": "Settings",
        "language": "Language",
        "summary_sentences": "Summary sentences",
        "num_keywords": "Number of keywords",
        "retrieved_passages": "Retrieved passages",
        "compare_snippets": "Evidence snippets per dimension",
        "sidebar_tip": "Tip: upload one or more papers. You can analyze each paper separately or compare them together.",
        "no_docs": "Upload a paper or enable the sample text to begin.",
        "documents": "Documents",
        "total_words": "Approx. words",
        "mode": "Mode",
        "multi_doc": "Multi-paper",
        "single_doc": "Single-paper",
        "summary_tab": "Summary",
        "keywords_tab": "Keywords",
        "ask_tab": "Ask / Search",
        "compare_tab": "Compare Papers",
        "domain_tab": "Domain Hints",
        "preview_tab": "Text Preview",
        "summary_header": "Grounded summary",
        "summary_caption": "Summaries are generated separately for each uploaded paper by default.",
        "summary_mode": "Summary mode",
        "separate_summaries": "Separate summaries",
        "combined_summary": "Combined summary",
        "combined_summary_warning": "Combined summary mixes all uploaded papers. Use it only when you want an overview of the full document set.",
        "keywords_header": "Keywords and phrases",
        "keywords_caption": "Keywords are shown separately by paper. You can also view combined keywords.",
        "keyword_mode": "Keyword mode",
        "by_paper": "By paper",
        "combined_keywords": "Combined keywords",
        "no_keywords_doc": "No keywords found for this document.",
        "no_keywords": "No keywords found. Try a longer document.",
        "ask_header": "Ask the paper",
        "ask_caption": "Ask in English or Chinese. You can target one paper or all uploaded papers.",
        "question_target": "Question target",
        "all_papers": "All uploaded papers",
        "example_questions": "Example questions",
        "your_question": "Your question",
        "submit_answer": "Generate grounded answer",
        "empty_question": "Please enter a question.",
        "show_evidence": "Show retrieved evidence passages",
        "evidence_passage": "Evidence passage",
        "source": "Source",
        "similarity": "similarity",
        "compare_header": "Compare papers",
        "compare_caption": "Compare research focus, methods, parameters, results, and limitations across papers.",
        "need_two_docs": "Please upload at least two documents to use comparison mode. You can also upload one paper and keep the sample text enabled for testing.",
        "core_table": "Core comparison table",
        "keyword_overlap": "Keyword overlap and distinctive keywords",
        "detailed_evidence": "Detailed evidence by dimension",
        "not_detected": "Not clearly detected.",
        "domain_header": "Materials-science domain hints",
        "domain_caption": "Domain hints are shown separately for each uploaded paper.",
        "no_hint": "No obvious hint detected.",
        "preview_header": "Document preview",
        "words": "words",
        "file_count": "file(s)",
    },
    "中文": {
        "app_title": "MatSci PaperLens AI",
        "app_subtitle": "面向材料科学与半导体论文的本地证据式 AI/NLP 阅读助手。",
        "hero_badge": "本地运行 · 基于原文证据 · 支持多论文",
        "upload_label": "上传一篇或多篇 PDF/TXT 论文",
        "sample_checkbox": "使用内置材料科学示例文本",
        "settings": "设置",
        "language": "语言",
        "summary_sentences": "摘要句子数量",
        "num_keywords": "关键词数量",
        "retrieved_passages": "检索证据段落数量",
        "compare_snippets": "每个对比维度的证据句数量",
        "sidebar_tip": "提示：可以上传一篇或多篇论文。既可以单篇分析，也可以多篇横向对比。",
        "no_docs": "请上传论文，或启用示例文本。",
        "documents": "文档数量",
        "total_words": "约总词数",
        "mode": "模式",
        "multi_doc": "多论文",
        "single_doc": "单论文",
        "summary_tab": "摘要",
        "keywords_tab": "关键词",
        "ask_tab": "问答 / 检索",
        "compare_tab": "论文对比",
        "domain_tab": "领域提示",
        "preview_tab": "文本预览",
        "summary_header": "证据式摘要",
        "summary_caption": "默认情况下，每篇上传论文会被单独总结，避免多篇内容混在一起。",
        "summary_mode": "摘要模式",
        "separate_summaries": "按论文分别总结",
        "combined_summary": "合并总结",
        "combined_summary_warning": "合并总结会混合所有上传论文。只有在想看整体概览时才建议使用。",
        "keywords_header": "关键词与短语",
        "keywords_caption": "默认按论文分别提取关键词，也可以查看所有论文的合并关键词。",
        "keyword_mode": "关键词模式",
        "by_paper": "按论文分别提取",
        "combined_keywords": "合并关键词",
        "no_keywords_doc": "这篇文档没有提取到明显关键词。",
        "no_keywords": "没有提取到关键词，请尝试上传更长或更清晰的文档。",
        "ask_header": "向论文提问",
        "ask_caption": "支持英文或中文提问。可以选择针对某一篇论文提问，也可以对所有上传论文提问。",
        "question_target": "提问对象",
        "all_papers": "所有上传论文",
        "example_questions": "示例问题",
        "your_question": "你的问题",
        "submit_answer": "生成基于原文证据的回答",
        "empty_question": "请输入一个问题。",
        "show_evidence": "显示检索到的证据段落",
        "evidence_passage": "证据段落",
        "source": "来源",
        "similarity": "相似度",
        "compare_header": "论文对比",
        "compare_caption": "对比多篇论文的研究重点、方法、参数、结果与局限。",
        "need_two_docs": "请至少上传两篇文档以使用论文对比功能。也可以上传一篇论文，同时保留示例文本用于测试。",
        "core_table": "核心对比表",
        "keyword_overlap": "共同关键词与特色关键词",
        "detailed_evidence": "按维度展开证据",
        "not_detected": "未明显检测到。",
        "domain_header": "材料科学领域提示",
        "domain_caption": "领域提示会按每篇论文分别展示。",
        "no_hint": "没有检测到明显提示。",
        "preview_header": "文本预览",
        "words": "词",
        "file_count": "个文件",
    },
}


def t(key: str) -> str:
    """Translate UI text based on selected language."""
    return UI_TEXT[st.session_state["language"]].get(key, key)


def inject_css() -> None:
    """Add lightweight custom CSS for a cleaner UI."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            border-radius: 1.2rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background: linear-gradient(135deg, rgba(245, 247, 250, 0.95), rgba(255, 255, 255, 0.95));
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            color: #262730;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #6b7280;
            margin-bottom: 0.75rem;
        }

        .hero-badge {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 75, 75, 0.10);
            color: #ff4b4b;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .section-card {
            padding: 1rem 1.15rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background: rgba(255, 255, 255, 0.72);
            margin-bottom: 1rem;
        }

        .doc-chip {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            margin: 0.15rem 0.22rem 0.15rem 0;
            border-radius: 999px;
            background: rgba(49, 51, 63, 0.06);
            color: #31333f;
            font-size: 0.82rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(250, 250, 252, 0.85);
            border: 1px solid rgba(49, 51, 63, 0.10);
            padding: 0.8rem 1rem;
            border-radius: 0.9rem;
        }

        div[data-testid="stExpander"] {
            border-radius: 0.9rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
        }

        .small-note {
            font-size: 0.88rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_uploaded_file(uploaded_file) -> str:
    """Read PDF or TXT content from a Streamlit uploaded file."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    return uploaded_file.read().decode("utf-8", errors="ignore")


def get_target_documents(
    target_index: int,
    documents: list[str],
    document_names: list[str],
):
    """Return selected document(s) based on user choice."""
    if target_index == 0:
        return documents, document_names

    selected_index = target_index - 1
    return [documents[selected_index]], [document_names[selected_index]]


def approx_word_count(text: str) -> int:
    """Approximate word count for English scientific text."""
    return len(text.split())


def render_doc_chips(document_names: list[str]) -> None:
    chips = "".join(
        f'<span class="doc-chip">{name}</span>'
        for name in document_names
    )
    st.markdown(chips, unsafe_allow_html=True)


def localize_analysis_markdown(markdown_text: str) -> str:
    """Lightly localize fixed analysis headings while preserving evidence text."""
    if st.session_state["language"] != "中文":
        return markdown_text

    replacements = {
        "### Grounded summary": "### 证据式摘要",
        "### Supporting points": "### 支撑证据点",
        "**Main focus:**": "**研究重点：**",
        "**Approach / method:**": "**方法 / 路径：**",
        "**Key result:**": "**关键结果：**",
        "**Limitation / future work:**": "**局限 / 未来工作：**",
        "### Grounded answer": "### 基于原文证据的回答",
        "**Detected question type:**": "**识别的问题类型：**",
        "**Confidence:**": "**置信度：**",
        "**Answer:**": "**回答：**",
        "**Evidence sources:**": "**证据来源：**",
        "High": "高",
        "Medium": "中",
        "Low": "低",
        "This is a grounded extractive summary.": "这是基于原文抽取的证据式摘要。",
        "The wording is lightly cleaned, but the factual content is selected from the original document.": "文字经过轻微清理，但事实内容来自原始文档。",
        "This QA mode is extractive and grounded.": "当前问答模式是基于原文抽取的证据式问答。",
        "It answers by selecting relevant sentences from the uploaded document, not by inventing new claims.": "它通过选择上传文档中的相关句子回答，而不是凭空生成新结论。",
    }

    for source, target in replacements.items():
        markdown_text = markdown_text.replace(source, target)

    return markdown_text


if "language" not in st.session_state:
    st.session_state["language"] = "English"


with st.sidebar:
    st.header("⚙️ Settings / 设置")

    language = st.selectbox(
        "Language / 语言",
        ["English", "中文"],
        index=0 if st.session_state["language"] == "English" else 1,
    )
    st.session_state["language"] = language

    max_summary_sentences = st.slider(t("summary_sentences"), 3, 10, 5)
    max_keywords = st.slider(t("num_keywords"), 5, 30, 15)
    top_k = st.slider(t("retrieved_passages"), 1, 8, 4)

    st.markdown("---")
    st.markdown(t("sidebar_tip"))


inject_css()


st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">🔬 {t("app_title")}</div>
        <div class="hero-subtitle">{t("app_subtitle")}</div>
        <span class="hero-badge">{t("hero_badge")}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_files = st.file_uploader(
    t("upload_label"),
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

use_sample = st.checkbox(t("sample_checkbox"), value=not uploaded_files)

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
    st.info(t("no_docs"))
    st.stop()

combined_text = "\n\n".join(documents)
total_words = sum(approx_word_count(doc) for doc in documents)

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

with metric_col_1:
    st.metric(t("documents"), len(documents))

with metric_col_2:
    st.metric(t("total_words"), f"{total_words:,}")

with metric_col_3:
    st.metric(t("mode"), t("multi_doc") if len(documents) > 1 else t("single_doc"))

render_doc_chips(document_names)

tabs = st.tabs(
    [
        t("summary_tab"),
        t("keywords_tab"),
        t("ask_tab"),
        t("compare_tab"),
        t("domain_tab"),
        t("preview_tab"),
    ]
)


with tabs[0]:
    st.subheader(t("summary_header"))
    st.caption(t("summary_caption"))

    summary_mode = st.radio(
        t("summary_mode"),
        [t("separate_summaries"), t("combined_summary")],
        horizontal=True,
    )

    if summary_mode == t("separate_summaries"):
        for doc_name, doc_text in zip(document_names, documents):
            with st.expander(doc_name, expanded=len(documents) <= 2):
                summary = summarize_text(
                    doc_text,
                    max_sentences=max_summary_sentences,
                )
                st.markdown(localize_analysis_markdown(summary))
    else:
        st.warning(t("combined_summary_warning"))
        summary = summarize_text(
            combined_text,
            max_sentences=max_summary_sentences,
        )
        st.markdown(localize_analysis_markdown(summary))


with tabs[1]:
    st.subheader(t("keywords_header"))
    st.caption(t("keywords_caption"))

    keyword_mode = st.radio(
        t("keyword_mode"),
        [t("by_paper"), t("combined_keywords")],
        horizontal=True,
    )

    if keyword_mode == t("by_paper"):
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
                    st.warning(t("no_keywords_doc"))
    else:
        keywords = extract_keywords(combined_text, top_n=max_keywords)

        if not keywords.empty:
            st.dataframe(
                keywords,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(t("no_keywords"))


with tabs[2]:
    st.subheader(t("ask_header"))
    st.caption(t("ask_caption"))

    target_options = [t("all_papers")] + document_names

    target_label = st.selectbox(
        t("question_target"),
        target_options,
        index=0,
    )

    target_index = target_options.index(target_label)

    if st.session_state["language"] == "中文":
        example_questions = [
            "这篇文章研究了什么材料体系？",
            "这篇文章用了什么方法或表征手段？",
            "这篇文章的主要结果是什么？",
            "这篇文章有什么局限或未来工作？",
            "这篇文章用了哪些参数或设定？",
            "What material system is studied?",
            "What methods or characterization techniques are used?",
        ]
    else:
        example_questions = [
            "What material system is studied?",
            "What methods or characterization techniques are used?",
            "What are the main results?",
            "What limitations or future work are mentioned?",
            "What parameters or settings are used?",
            "这篇文章研究了什么材料体系？",
            "这篇文章用了什么方法或表征手段？",
        ]

    selected_example = st.selectbox(
        t("example_questions"),
        example_questions,
        index=0,
    )

    with st.form("qa_form"):
        query = st.text_area(
            t("your_question"),
            value=selected_example,
            height=110,
        )

        submitted = st.form_submit_button(t("submit_answer"))

    if submitted:
        if not query.strip():
            st.warning(t("empty_question"))
        else:
            target_documents, target_document_names = get_target_documents(
                target_index,
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

            st.markdown(localize_analysis_markdown(qa_result["answer"]))

            evidence = qa_result.get("evidence", [])

            if evidence:
                with st.expander(t("show_evidence"), expanded=False):
                    for i, item in enumerate(evidence, start=1):
                        source_index = item["document_index"]
                        source_name = target_document_names[source_index]
                        st.markdown(f"#### {t('evidence_passage')} {i}")
                        st.caption(
                            f"{t('source')}: {source_name} | {t('similarity')}: {item['score']:.3f}"
                        )
                        st.write(item["passage"])


with tabs[3]:
    st.subheader(t("compare_header"))
    st.caption(t("compare_caption"))

    if len(documents) < 2:
        st.info(t("need_two_docs"))
    else:
        max_compare_snippets = st.slider(
            t("compare_snippets"),
            min_value=1,
            max_value=5,
            value=3,
        )

        comparison = compare_documents(
            documents=documents,
            document_names=document_names,
            max_snippets_per_dimension=max_compare_snippets,
        )

        st.markdown(f"### {t('core_table')}")
        st.dataframe(
            comparison["table"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(f"### {t('keyword_overlap')}")
        st.dataframe(
            comparison["keyword_table"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(f"### {t('detailed_evidence')}")

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
                            st.caption(t("not_detected"))

        st.info(comparison["note"])


with tabs[4]:
    st.subheader(t("domain_header"))
    st.caption(t("domain_caption"))

    for doc_name, doc_text in zip(document_names, documents):
        with st.expander(doc_name, expanded=len(documents) == 1):
            hints = find_domain_hints(doc_text)

            for category, snippets in hints.items():
                st.markdown(f"### {category.replace('_', ' ').title()}")

                if snippets:
                    for snippet in snippets:
                        st.markdown(f"- {snippet}")
                else:
                    st.caption(t("no_hint"))


with tabs[5]:
    st.subheader(t("preview_header"))

    for name, doc in zip(document_names, documents):
        with st.expander(name, expanded=False):
            st.caption(f"{approx_word_count(doc):,} {t('words')}")
            st.write(doc[:5000] + ("..." if len(doc) > 5000 else ""))
