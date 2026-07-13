import pytest

from src.report_utils import build_markdown_report


def test_markdown_report_keeps_documents_separate_and_includes_sources():
    report = build_markdown_report(
        [
            "Raman spectroscopy characterizes the transferred ferroelectric layer.",
            "X-ray diffraction characterizes the annealed semiconductor film.",
        ],
        ["paper_a.pdf", "paper_b.pdf"],
        max_summary_sentences=2,
        top_keywords=3,
    )

    assert "## paper_a.pdf" in report
    assert "## paper_b.pdf" in report
    assert report.index("## paper_a.pdf") < report.index("## paper_b.pdf")
    assert "### Keywords" in report
    assert "### Domain evidence" in report


def test_markdown_report_rejects_mismatched_document_metadata():
    with pytest.raises(ValueError):
        build_markdown_report(["One complete document sentence."], [])


def test_markdown_report_includes_saved_grounded_questions():
    report = build_markdown_report(
        ["Raman spectroscopy characterizes the ferroelectric film."],
        ["paper.pdf"],
        qa_history=[
            {
                "question": "Which method is used?",
                "target": "paper.pdf",
                "answer": "### Grounded answer\n\nRaman spectroscopy is used. [E1]",
            }
        ],
    )

    assert "## Questions and grounded answers" in report
    assert "### Q1. Which method is used?" in report
    assert "**Target:** paper.pdf" in report
    assert "Raman spectroscopy is used. [E1]" in report


def test_markdown_report_preserves_pdf_page_citations():
    pages = [
        "This work studies a ferroelectric thin film.",
        "Raman spectroscopy characterizes the transferred layer.",
    ]
    report = build_markdown_report(
        ["\n\n".join(pages)],
        ["paper.pdf"],
        max_summary_sentences=2,
        document_pages=[pages],
    )

    assert "[p. 1]" in report
    assert "[p. 2]" in report
