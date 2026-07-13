from __future__ import annotations

from typing import Sequence

from src.nlp_pipeline import extract_keywords, find_domain_hints, summarize_text


def build_markdown_report(
    documents: Sequence[str],
    document_names: Sequence[str],
    max_summary_sentences: int = 5,
    top_keywords: int = 15,
    qa_history: Sequence[dict[str, object]] | None = None,
    document_pages: Sequence[Sequence[str]] | None = None,
) -> str:
    """Build a portable, source-separated Markdown report."""
    if len(documents) != len(document_names):
        raise ValueError("documents and document_names must have the same length")

    lines = [
        "# MatSci PaperLens AI report",
        "",
        "> Local, extractive analysis. Verify all evidence against the original papers.",
        "",
        f"Documents analyzed: {len(documents)}",
    ]

    for document_index, (document, name) in enumerate(zip(documents, document_names)):
        page_texts = (
            document_pages[document_index]
            if document_pages and document_index < len(document_pages)
            else None
        )
        lines.extend(["", "---", "", f"## {name}", ""])
        lines.append(
            summarize_text(
                document,
                max_sentences=max_summary_sentences,
                page_texts=page_texts,
            )
        )

        keyword_frame = extract_keywords(document, top_n=top_keywords)
        lines.extend(["", "### Keywords", ""])
        if keyword_frame.empty:
            lines.append("No keywords were detected.")
        else:
            lines.append(
                ", ".join(keyword_frame["keyword"].astype(str).tolist())
            )

        lines.extend(["", "### Domain evidence", ""])
        hints = find_domain_hints(document, page_texts=page_texts)
        detected_any = False
        for category, snippets in hints.items():
            if not snippets:
                continue
            detected_any = True
            label = category.replace("_", " ").title()
            lines.append(f"#### {label}")
            lines.extend(f"- {snippet}" for snippet in snippets)
            lines.append("")

        if not detected_any:
            lines.append("No domain-specific evidence was clearly detected.")

    if qa_history:
        lines.extend(["", "---", "", "## Questions and grounded answers", ""])
        for index, item in enumerate(qa_history, start=1):
            question = str(item.get("question", "")).strip()
            target = str(item.get("target", "All papers")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            answer = answer.replace("### Grounded answer", "#### Grounded answer", 1)
            answer = answer.replace("### 基于原文证据的回答", "#### 基于原文证据的回答", 1)
            lines.extend(
                [
                    f"### Q{index}. {question}",
                    "",
                    f"**Target:** {target}",
                    "",
                    answer,
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"
