from src.document_utils import make_unique_document_name


def test_duplicate_document_names_get_readable_suffixes():
    existing = ["paper.pdf", "paper (2).pdf"]

    assert make_unique_document_name("paper.pdf", existing) == "paper (3).pdf"
    assert make_unique_document_name("different.pdf", existing) == "different.pdf"
