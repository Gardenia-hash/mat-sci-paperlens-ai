from pathlib import Path
from typing import BinaryIO, Union

import fitz


def extract_page_texts_from_pdf(file_obj: Union[str, Path, BinaryIO]) -> list[str]:
    """Extract text page by page while preserving physical PDF page order.

    Parameters
    ----------
    file_obj:
        A filesystem path or a file-like object, such as Streamlit's UploadedFile.

    Empty pages remain in the returned list so list position always maps to the
    physical PDF page number (index 0 is page 1).
    """
    if isinstance(file_obj, (str, Path)):
        doc = fitz.open(str(file_obj))
    else:
        try:
            file_obj.seek(0)
        except Exception:
            pass
        pdf_bytes = file_obj.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text.strip() if text else "")

    doc.close()
    return pages


def extract_text_from_pdf(file_obj: Union[str, Path, BinaryIO]) -> str:
    """Extract plain text from all PDF pages.

    Use :func:`extract_page_texts_from_pdf` when downstream analysis needs
    physical page citations.
    """
    return "\n\n".join(
        page_text for page_text in extract_page_texts_from_pdf(file_obj) if page_text
    )
