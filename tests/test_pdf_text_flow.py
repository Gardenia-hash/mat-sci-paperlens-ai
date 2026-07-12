from io import BytesIO

import fitz

from src.pdf_utils import extract_text_from_pdf
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
