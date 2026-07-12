from src.document_utils import make_unique_document_name, upload_limit_violation


def test_duplicate_document_names_get_readable_suffixes():
    existing = ["paper.pdf", "paper (2).pdf"]

    assert make_unique_document_name("paper.pdf", existing) == "paper (3).pdf"
    assert make_unique_document_name("different.pdf", existing) == "different.pdf"


def test_upload_limit_violation_checks_count_and_total_size():
    assert upload_limit_violation([10, 20], max_files=3, max_total_bytes=40) is None
    assert (
        upload_limit_violation([10, 20, 30], max_files=2, max_total_bytes=100)
        == "too_many_files"
    )
    assert (
        upload_limit_violation([30, 30], max_files=3, max_total_bytes=50)
        == "total_size_exceeded"
    )
