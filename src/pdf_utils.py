from pathlib import Path
from typing import BinaryIO, Union

import fitz


def extract_text_from_pdf(file_obj: Union[str, Path, BinaryIO]) -> str:
    """Extract plain text from a PDF path or file-like object.

    Parameters
    ----------
    file_obj:
        A filesystem path or a file-like object, such as Streamlit's UploadedFile.

    Returns
    -------
    str
        Extracted text from all pages.
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

    pages = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text.strip())

    doc.close()
    return "\n\n".join(pages)
