from io import BytesIO

import fitz

from src.pdf_utils import extract_page_texts_from_pdf, extract_text_from_pdf
from src.text_utils import split_sentences


def test_pdf_extraction_hard_wraps_are_reconstructed_before_analysis():
    document = fitz.open()
    page = document.new_page(width=500, height=500)
    page.insert_text(
        (50, 80),
        "We demonstrate a stable ferroelectric device\n"
        "using a scalable dry-transfer process\n"
        "that improves switching uniformity across the sample.\n"
        "The optical response remains stable after repeated cycling.",
        fontsize=10,
    )
    pdf_bytes = document.tobytes()
    document.close()

    extracted = extract_text_from_pdf(BytesIO(pdf_bytes))
    sentences = split_sentences(extracted)

    assert sentences[0] == (
        "We demonstrate a stable ferroelectric device using a scalable dry-transfer "
        "process that improves switching uniformity across the sample."
    )
    assert sentences[1] == "The optical response remains stable after repeated cycling."


def test_pdf_page_extraction_preserves_empty_pages_and_page_numbers():
    document = fitz.open()
    first_page = document.new_page(width=500, height=500)
    first_page.insert_text((50, 80), "The film is deposited by ALD.", fontsize=10)
    document.new_page(width=500, height=500)
    third_page = document.new_page(width=500, height=500)
    third_page.insert_text((50, 80), "Raman spectroscopy confirms the phase.", fontsize=10)
    pdf_bytes = document.tobytes()
    document.close()

    pages = extract_page_texts_from_pdf(BytesIO(pdf_bytes))

    assert len(pages) == 3
    assert "deposited by ALD" in pages[0]
    assert pages[1] == ""
    assert "Raman spectroscopy" in pages[2]
