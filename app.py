from io import BytesIO
from html import escape
from pathlib import Path

import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.figure_utils import explain_figure, extract_figures_from_pdf
from src.document_utils import make_unique_document_name
from src.nlp_pipeline import (
    answer_question,
    compare_documents,
    extract_keywords,
    summarize_text,
    find_domain_hints,
)
from src.report_utils import build_markdown_report
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
        "benefit_private": "Private by default",
        "benefit_private_detail": "No API key required",
        "benefit_grounded": "Evidence first",
        "benefit_grounded_detail": "Sources stay visible",
        "benefit_compare": "Built for research",
        "benefit_compare_detail": "Read and compare papers",
        "step_upload": "1 · Add papers",
        "step_upload_detail": "Upload PDF, TXT, or Markdown",
        "step_explore": "2 · Explore evidence",
        "step_explore_detail": "Summarize, ask, and compare",
        "step_export": "3 · Keep your work",
        "step_export_detail": "Download a Markdown report",
        "upload_label": "Upload one or more PDF, TXT, or Markdown papers",
        "sample_checkbox": "Try the built-in demo (no upload needed)",
        "settings": "Settings",
        "language": "Language",
        "analysis_controls": "Analysis controls",
        "reset_workspace": "Reset workspace",
        "star_project": "Star this project on GitHub",
        "star_note": "Finding it useful? A GitHub star helps other researchers discover it.",
        "processing_documents": "Reading uploaded papers…",
        "processing_file": "Reading",
        "processing_complete": "Papers are ready",
        "processing_partial": "Some files could not be read",
        "file_errors": "Show file errors",
        "workspace_ready": "Workspace ready · Papers: {documents} · Figures: {figures}. Choose a tool below.",
        "summary_sentences": "Summary sentences",
        "num_keywords": "Number of keywords",
        "retrieved_passages": "Retrieved passages",
        "compare_snippets": "Evidence snippets per dimension",
        "sidebar_tip": "Tip: upload one or more papers. You can analyze each paper separately or compare them together.",
        "no_docs": "Upload a paper or enable the sample text to begin.",
        "documents": "Documents",
        "figures_metric": "Figures",
        "total_words": "Approx. words",
        "mode": "Mode",
        "multi_doc": "Multi-paper",
        "single_doc": "Single-paper",
        "summary_tab": "Summary",
        "keywords_tab": "Keywords",
        "ask_tab": "Ask / Search",
        "compare_tab": "Compare Papers",
        "domain_tab": "Domain Hints",
        "figures_tab": "Figures",
        "export_tab": "Export",
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
        "figures_header": "Figures and grounded interpretation",
        "figures_caption": "Raster figures are extracted locally. The baseline explanation uses only captions and nearby text; it does not guess from pixels.",
        "figure_document": "Paper containing the figure",
        "figure_select": "Select a figure",
        "figure_source": "Figure source",
        "figure_page": "Page",
        "figure_size": "Extracted size",
        "original_caption": "Original caption",
        "missing_caption": "No caption was detected near this image.",
        "system_explanation": "System interpretation",
        "direct_evidence": "Direct evidence",
        "reasonable_inference": "Reasonable inference",
        "unknown": "Unknown",
        "no_figures": "No meaningful raster figures were extracted. The PDF may be scanned, vector-only, or contain only small decorative images.",
        "no_figures_document": "No meaningful raster figures were extracted from this paper.",
        "export_header": "Export your research notes",
        "export_caption": "Create a source-separated Markdown report that you can keep, edit, or share.",
        "export_includes": "Includes grounded summaries, keywords, and domain evidence for every paper.",
        "prepare_report": "Prepare Markdown report",
        "preparing_report": "Building your report…",
        "download_report": "Download report",
        "report_ready": "Your report is ready.",
        "report_preview": "Preview report",
        "preview_header": "Document preview",
        "words": "words",
        "file_count": "file(s)",
    },
    "中文": {
        "app_title": "MatSci PaperLens AI",
        "app_subtitle": "面向材料科学与半导体论文的本地证据式 AI/NLP 阅读助手。",
        "hero_badge": "本地运行 · 基于原文证据 · 支持多论文",
        "benefit_private": "默认保护隐私",
        "benefit_private_detail": "无需 API Key",
        "benefit_grounded": "证据优先",
        "benefit_grounded_detail": "始终显示来源",
        "benefit_compare": "为科研而设计",
        "benefit_compare_detail": "阅读并对比多篇论文",
        "step_upload": "1 · 添加论文",
        "step_upload_detail": "上传 PDF、TXT 或 Markdown",
        "step_explore": "2 · 探索证据",
        "step_explore_detail": "摘要、问答与对比",
        "step_export": "3 · 保存成果",
        "step_export_detail": "下载 Markdown 报告",
        "upload_label": "上传一篇或多篇 PDF、TXT 或 Markdown 论文",
        "sample_checkbox": "体验内置示例（无需上传）",
        "settings": "设置",
        "language": "语言",
        "analysis_controls": "分析参数",
        "reset_workspace": "重置工作区",
        "star_project": "在 GitHub 为项目加星",
        "star_note": "如果它对你有帮助，一个 GitHub Star 能让更多研究者发现它。",
        "processing_documents": "正在读取上传的论文…",
        "processing_file": "正在读取",
        "processing_complete": "论文已准备完成",
        "processing_partial": "部分文件无法读取",
        "file_errors": "查看文件错误",
        "workspace_ready": "工作区已就绪 · 论文：{documents} · 图像：{figures}。请选择下方工具。",
        "summary_sentences": "摘要句子数量",
        "num_keywords": "关键词数量",
        "retrieved_passages": "检索证据段落数量",
        "compare_snippets": "每个对比维度的证据句数量",
        "sidebar_tip": "提示：可以上传一篇或多篇论文。既可以单篇分析，也可以多篇横向对比。",
        "no_docs": "请上传论文，或启用示例文本。",
        "documents": "文档数量",
        "figures_metric": "图像数量",
        "total_words": "约总词数",
        "mode": "模式",
        "multi_doc": "多论文",
        "single_doc": "单论文",
        "summary_tab": "摘要",
        "keywords_tab": "关键词",
        "ask_tab": "问答 / 检索",
        "compare_tab": "论文对比",
        "domain_tab": "领域提示",
        "figures_tab": "图像解读",
        "export_tab": "导出",
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
        "figures_header": "论文图像与证据式解读",
        "figures_caption": "图像在本地提取。基础解释只使用原文 caption 和同页邻近文字，不根据像素猜测结论。",
        "figure_document": "选择图像所属论文",
        "figure_select": "选择图像",
        "figure_source": "图像来源",
        "figure_page": "页码",
        "figure_size": "提取尺寸",
        "original_caption": "原文 caption",
        "missing_caption": "该图附近未检测到 caption。",
        "system_explanation": "系统解释",
        "direct_evidence": "直接证据",
        "reasonable_inference": "合理推断",
        "unknown": "无法确认",
        "no_figures": "没有提取到有效的栅格图像。PDF 可能是扫描件、只含矢量图，或只含很小的装饰图片。",
        "no_figures_document": "这篇论文没有提取到有效的栅格图像。",
        "export_header": "导出研究笔记",
        "export_caption": "生成按来源分隔的 Markdown 报告，方便保存、编辑或分享。",
        "export_includes": "报告包含每篇论文的证据式摘要、关键词和领域证据。",
        "prepare_report": "生成 Markdown 报告",
        "preparing_report": "正在生成报告…",
        "download_report": "下载报告",
        "report_ready": "报告已准备完成。",
        "report_preview": "预览报告",
        "preview_header": "文本预览",
        "words": "词",
        "file_count": "个文件",
    },
}


def t(key: str) -> str:
    """Translate UI text based on selected language."""
    return UI_TEXT[st.session_state["language"]].get(key, key)


@st.cache_data(show_spinner=False)
def process_pdf_upload(file_bytes: bytes, document_name: str):
    """Parse one PDF once and reuse the result across Streamlit reruns."""
    text = clean_text(extract_text_from_pdf(BytesIO(file_bytes)))
    figures = extract_figures_from_pdf(file_bytes, document_name=document_name)
    return text, figures


@st.cache_data(show_spinner=False)
def cached_summary(text: str, max_sentences: int) -> str:
    return summarize_text(text, max_sentences=max_sentences)


@st.cache_data(show_spinner=False)
def cached_keywords(text: str, top_n: int):
    return extract_keywords(text, top_n=top_n)


@st.cache_data(show_spinner=False)
def cached_answer(
    documents: tuple[str, ...],
    query: str,
    document_names: tuple[str, ...],
    top_k: int,
):
    return answer_question(
        documents=list(documents),
        query=query,
        document_names=list(document_names),
        top_k=top_k,
        max_answer_sentences=4,
    )


@st.cache_data(show_spinner=False)
def cached_comparison(
    documents: tuple[str, ...],
    document_names: tuple[str, ...],
    max_snippets: int,
):
    return compare_documents(
        documents=list(documents),
        document_names=list(document_names),
        max_snippets_per_dimension=max_snippets,
    )


@st.cache_data(show_spinner=False)
def cached_domain_hints(text: str):
    return find_domain_hints(text)


@st.cache_data(show_spinner=False)
def cached_markdown_report(
    documents: tuple[str, ...],
    document_names: tuple[str, ...],
    max_summary_sentences: int,
    max_keywords: int,
) -> str:
    return build_markdown_report(
        documents,
        document_names,
        max_summary_sentences=max_summary_sentences,
        top_keywords=max_keywords,
    )


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

        .benefit-grid, .step-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.95rem 0 1.15rem 0;
        }

        .benefit-item, .step-item {
            padding: 0.8rem 0.9rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(49, 51, 63, 0.10);
            background: rgba(250, 250, 252, 0.82);
        }

        .benefit-title, .step-title {
            color: #262730;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }

        .benefit-detail, .step-detail {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.35;
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

        @media (max-width: 760px) {
            .benefit-grid, .step-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        f'<span class="doc-chip">{escape(name)}</span>'
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
if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0
if "use_sample" not in st.session_state:
    st.session_state["use_sample"] = True
if "last_upload_signature" not in st.session_state:
    st.session_state["last_upload_signature"] = ()


with st.sidebar:
    st.header("⚙️ Settings / 设置")

    language = st.selectbox(
        "Language / 语言",
        ["English", "中文"],
        index=0 if st.session_state["language"] == "English" else 1,
    )
    st.session_state["language"] = language

    with st.expander(t("analysis_controls"), expanded=False):
        max_summary_sentences = st.slider(t("summary_sentences"), 3, 10, 5)
        max_keywords = st.slider(t("num_keywords"), 5, 30, 15)
        top_k = st.slider(t("retrieved_passages"), 1, 8, 4)

    if st.button(t("reset_workspace"), use_container_width=True):
        st.session_state["uploader_version"] += 1
        st.session_state["use_sample"] = True
        st.session_state["last_upload_signature"] = ()
        st.session_state.pop("report_content", None)
        st.session_state.pop("report_signature", None)
        st.rerun()

    st.markdown("---")
    st.markdown(t("sidebar_tip"))
    st.caption(t("star_note"))
    st.link_button(
        f"⭐ {t('star_project')}",
        "https://github.com/Gardenia-hash/mat-sci-paperlens-ai",
        use_container_width=True,
    )


inject_css()


st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">🔬 {t("app_title")}</div>
        <div class="hero-subtitle">{t("app_subtitle")}</div>
        <span class="hero-badge">{t("hero_badge")}</span>
        <div class="benefit-grid">
            <div class="benefit-item">
                <div class="benefit-title">🔒 {t("benefit_private")}</div>
                <div class="benefit-detail">{t("benefit_private_detail")}</div>
            </div>
            <div class="benefit-item">
                <div class="benefit-title">📎 {t("benefit_grounded")}</div>
                <div class="benefit-detail">{t("benefit_grounded_detail")}</div>
            </div>
            <div class="benefit-item">
                <div class="benefit-title">🧪 {t("benefit_compare")}</div>
                <div class="benefit-detail">{t("benefit_compare_detail")}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="step-grid">
        <div class="step-item">
            <div class="step-title">{t("step_upload")}</div>
            <div class="step-detail">{t("step_upload_detail")}</div>
        </div>
        <div class="step-item">
            <div class="step-title">{t("step_explore")}</div>
            <div class="step-detail">{t("step_explore_detail")}</div>
        </div>
        <div class="step-item">
            <div class="step-title">{t("step_export")}</div>
            <div class="step-detail">{t("step_export_detail")}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


uploaded_files = st.file_uploader(
    t("upload_label"),
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    key=f"paper_uploader_{st.session_state['uploader_version']}",
)

upload_signature = tuple(
    (uploaded_file.name, uploaded_file.size)
    for uploaded_file in (uploaded_files or [])
)
if upload_signature and upload_signature != st.session_state["last_upload_signature"]:
    st.session_state["use_sample"] = False
    st.session_state["last_upload_signature"] = upload_signature

use_sample = st.checkbox(t("sample_checkbox"), key="use_sample")

documents = []
document_names = []
figures_by_document = {}
processing_errors = []

if uploaded_files:
    with st.status(t("processing_documents"), expanded=False) as status:
        for uploaded_file in uploaded_files:
            display_name = make_unique_document_name(
                uploaded_file.name,
                document_names,
            )
            status.write(f"{t('processing_file')}: {display_name}")

            try:
                file_bytes = uploaded_file.getvalue()
                if uploaded_file.name.lower().endswith(".pdf"):
                    text, figures = process_pdf_upload(file_bytes, display_name)
                else:
                    text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
                    figures = []

                if text or figures:
                    documents.append(text)
                    document_names.append(display_name)
                    figures_by_document[display_name] = figures
                else:
                    processing_errors.append(
                        f"{display_name}: no readable text or raster figures were found."
                    )
            except Exception as exc:
                processing_errors.append(f"{display_name}: {exc}")

        status.update(
            label=(
                t("processing_partial")
                if processing_errors
                else t("processing_complete")
            ),
            state="error" if processing_errors else "complete",
            expanded=bool(processing_errors),
        )

if processing_errors:
    with st.expander(t("file_errors"), expanded=True):
        for error in processing_errors:
            st.error(error)

if use_sample:
    sample_path = Path("data/sample_materials_text.txt")
    if sample_path.exists():
        sample_text = clean_text(sample_path.read_text(encoding="utf-8"))
        sample_name = make_unique_document_name(
            "sample_materials_text.txt",
            document_names,
        )
        documents.append(sample_text)
        document_names.append(sample_name)
        figures_by_document[sample_name] = []

if not documents:
    st.info(t("no_docs"))
    st.stop()

combined_text = "\n\n".join(documents)
total_words = sum(approx_word_count(doc) for doc in documents)
total_figures = sum(len(items) for items in figures_by_document.values())

st.success(
    t("workspace_ready").format(
        documents=len(documents),
        figures=total_figures,
    )
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

with metric_col_1:
    st.metric(t("documents"), len(documents))

with metric_col_2:
    st.metric(t("total_words"), f"{total_words:,}")

with metric_col_3:
    st.metric(t("figures_metric"), total_figures)

with metric_col_4:
    st.metric(t("mode"), t("multi_doc") if len(documents) > 1 else t("single_doc"))

render_doc_chips(document_names)

tabs = st.tabs(
    [
        t("summary_tab"),
        t("keywords_tab"),
        t("ask_tab"),
        t("compare_tab"),
        t("domain_tab"),
        t("figures_tab"),
        t("export_tab"),
        t("preview_tab"),
    ]
)


with tabs[0]:
    st.subheader(t("summary_header"))
    st.caption(t("summary_caption"))

    if len(documents) > 1:
        summary_mode = st.radio(
            t("summary_mode"),
            [t("separate_summaries"), t("combined_summary")],
            horizontal=True,
        )
    else:
        summary_mode = t("separate_summaries")

    if summary_mode == t("separate_summaries"):
        for doc_name, doc_text in zip(document_names, documents):
            with st.expander(doc_name, expanded=len(documents) <= 2):
                summary = cached_summary(
                    doc_text,
                    max_summary_sentences,
                )
                st.markdown(localize_analysis_markdown(summary))
    else:
        st.warning(t("combined_summary_warning"))
        summary = cached_summary(
            combined_text,
            max_summary_sentences,
        )
        st.markdown(localize_analysis_markdown(summary))


with tabs[1]:
    st.subheader(t("keywords_header"))
    st.caption(t("keywords_caption"))

    if len(documents) > 1:
        keyword_mode = st.radio(
            t("keyword_mode"),
            [t("by_paper"), t("combined_keywords")],
            horizontal=True,
        )
    else:
        keyword_mode = t("by_paper")

    if keyword_mode == t("by_paper"):
        for doc_name, doc_text in zip(document_names, documents):
            with st.expander(doc_name, expanded=len(documents) <= 2):
                keywords = cached_keywords(doc_text, max_keywords)

                if not keywords.empty:
                    st.dataframe(
                        keywords,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning(t("no_keywords_doc"))
    else:
        keywords = cached_keywords(combined_text, max_keywords)

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

    if len(documents) > 1:
        target_options = [t("all_papers")] + document_names
        target_label = st.selectbox(
            t("question_target"),
            target_options,
            index=0,
        )
        target_index = target_options.index(target_label)
    else:
        target_index = 1

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

            qa_result = cached_answer(
                tuple(target_documents),
                query,
                tuple(target_document_names),
                top_k,
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

        comparison = cached_comparison(
            tuple(documents),
            tuple(document_names),
            max_compare_snippets,
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
            hints = cached_domain_hints(doc_text)

            for category, snippets in hints.items():
                st.markdown(f"### {category.replace('_', ' ').title()}")

                if snippets:
                    for snippet in snippets:
                        st.markdown(f"- {snippet}")
                else:
                    st.caption(t("no_hint"))


with tabs[5]:
    st.subheader(t("figures_header"))
    st.caption(t("figures_caption"))

    if total_figures == 0:
        st.info(t("no_figures"))
    else:
        if len(document_names) > 1:
            figure_document = st.selectbox(
                t("figure_document"),
                document_names,
                key="figure_document",
            )
        else:
            figure_document = document_names[0]
        available_figures = figures_by_document.get(figure_document, [])

        if not available_figures:
            st.info(t("no_figures_document"))
        else:
            figure_labels = [
                f"Figure {figure.image_index} · {t('figure_page')} {figure.page_number}"
                for figure in available_figures
            ]
            selected_label = st.selectbox(
                t("figure_select"),
                figure_labels,
                key="figure_selector",
            )
            figure = available_figures[figure_labels.index(selected_label)]

            st.image(
                figure.image_bytes,
                caption=f"{t('figure_source')}: {figure.document_name} · {t('figure_page')} {figure.page_number}",
                use_container_width=True,
            )

            meta_1, meta_2, meta_3 = st.columns(3)
            meta_1.metric(t("figure_page"), figure.page_number)
            meta_2.metric(t("figure_size"), f"{figure.width} × {figure.height}")
            meta_3.metric("Format", figure.extension.upper())

            st.markdown(f"### {t('original_caption')}")
            if figure.caption:
                st.write(figure.caption)
            else:
                st.warning(t("missing_caption"))

            st.markdown(f"### {t('system_explanation')}")
            explanation = explain_figure(
                figure,
                language=st.session_state["language"],
            )
            for heading, items in (
                (t("direct_evidence"), explanation.direct_evidence),
                (t("reasonable_inference"), explanation.reasonable_inference),
                (t("unknown"), explanation.unknown),
            ):
                st.markdown(f"#### {heading}")
                if items:
                    for item in items:
                        st.markdown(f"- {item}")
                else:
                    st.caption(t("not_detected"))


with tabs[6]:
    st.subheader(t("export_header"))
    st.caption(t("export_caption"))
    st.info(t("export_includes"))

    workspace_signature = tuple(
        (name, len(document), hash(document))
        for name, document in zip(document_names, documents)
    )

    if st.button(t("prepare_report"), type="primary"):
        with st.spinner(t("preparing_report")):
            st.session_state["report_content"] = cached_markdown_report(
                tuple(documents),
                tuple(document_names),
                max_summary_sentences,
                max_keywords,
            )
            st.session_state["report_signature"] = workspace_signature

    report_content = st.session_state.get("report_content")
    if (
        report_content
        and st.session_state.get("report_signature") == workspace_signature
    ):
        st.success(t("report_ready"))
        st.download_button(
            t("download_report"),
            data=report_content,
            file_name="matsci-paperlens-report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        with st.expander(t("report_preview"), expanded=False):
            st.markdown(report_content)


with tabs[7]:
    st.subheader(t("preview_header"))

    for name, doc in zip(document_names, documents):
        with st.expander(name, expanded=False):
            st.caption(f"{approx_word_count(doc):,} {t('words')}")
            st.write(doc[:5000] + ("..." if len(doc) > 5000 else ""))
