import base64
from io import BytesIO

import fitz

from src.figure_utils import explain_figure, extract_figures_from_pdf


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def make_pdf(caption: str) -> bytes:
    document = fitz.open()
    page = document.new_page(width=400, height=500)
    page.insert_image(fitz.Rect(50, 80, 350, 260), stream=ONE_PIXEL_PNG)
    page.insert_text((50, 290), caption, fontsize=10)
    page.insert_text(
        (50, 330),
        "The main text discusses the measurement shown in Figure 1.",
        fontsize=9,
    )
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_figures_records_source_page_and_caption():
    figures = extract_figures_from_pdf(
        make_pdf("Figure 1. Spectrum comparison for two samples."),
        document_name="paper-a.pdf",
        min_width=1,
        min_height=1,
        min_area=1,
    )

    assert len(figures) == 1
    figure = figures[0]
    assert figure.document_name == "paper-a.pdf"
    assert figure.page_number == 1
    assert figure.image_index == 1
    assert figure.caption.startswith("Figure 1.")
    assert figure.image_bytes


def test_extract_figures_preserves_stream_position_and_filters_tiny_images():
    stream = BytesIO(make_pdf("Fig. 1. Test image."))
    stream.seek(7)

    figures = extract_figures_from_pdf(stream, document_name="paper.pdf")

    assert figures == []
    assert stream.tell() == 7


def test_figures_from_multiple_documents_remain_isolated():
    first = extract_figures_from_pdf(
        make_pdf("Figure 1. First paper."),
        document_name="first.pdf",
        min_width=1,
        min_height=1,
        min_area=1,
    )[0]
    second = extract_figures_from_pdf(
        make_pdf("Figure 1. Second paper."),
        document_name="second.pdf",
        min_width=1,
        min_height=1,
        min_area=1,
    )[0]

    assert first.document_id != second.document_id
    assert first.document_name == "first.pdf"
    assert second.document_name == "second.pdf"
    assert "Second paper" not in first.caption


def test_explanation_separates_evidence_inference_and_unknown():
    figure = extract_figures_from_pdf(
        make_pdf("Figure 1. Spectrum comparison for two samples."),
        document_name="paper.pdf",
        min_width=1,
        min_height=1,
        min_area=1,
    )[0]

    explanation = explain_figure(figure)

    assert explanation.direct_evidence
    assert explanation.reasonable_inference
    assert explanation.unknown
    assert any("exact values" in item for item in explanation.unknown)
