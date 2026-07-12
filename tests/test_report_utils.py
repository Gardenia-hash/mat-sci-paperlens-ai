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
