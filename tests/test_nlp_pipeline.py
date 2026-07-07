from src.nlp_pipeline import (
    answer_question,
    compare_documents,
    extract_keywords,
    retrieve_passages,
    summarize_text,
    find_domain_hints,
)


SAMPLE_TEXT = """
Van der Waals ferroelectric materials are promising for compact nonlinear optical devices.
The sample is prepared by exfoliation and dry transfer onto a silicon substrate.
Scanning electron microscopy and Raman spectroscopy are used for characterization.
The main result is a polarization-dependent second-harmonic generation signal.
However, the device is limited by contamination and nonuniform switching.
"""


def test_summarize_text_returns_text():
    summary = summarize_text(SAMPLE_TEXT, max_sentences=2)
    assert isinstance(summary, str)
    assert len(summary) > 20


def test_extract_keywords_returns_dataframe():
    keywords = extract_keywords(SAMPLE_TEXT, top_n=5)
    assert list(keywords.columns) == ["keyword", "score"]
    assert len(keywords) > 0


def test_retrieve_passages_returns_relevant_result():
    results = retrieve_passages([SAMPLE_TEXT], query="What characterization methods are used?", top_k=2)
    assert len(results) > 0
    assert "passage" in results[0]
    assert "score" in results[0]


def test_find_domain_hints_has_categories():
    hints = find_domain_hints(SAMPLE_TEXT)
    assert "materials_system" in hints
    assert "characterization" in hints

def test_answer_question_returns_grounded_answer():
    result = answer_question(
        [SAMPLE_TEXT],
        query="What characterization methods are used?",
        document_names=["sample"],
        top_k=2,
    )

    assert "answer" in result
    assert "evidence" in result
    assert "Grounded answer" in result["answer"]

def test_compare_documents_returns_table():
    doc_a = """
    This work studies a van der Waals ferroelectric thin film.
    The sample is prepared by exfoliation and dry transfer.
    Raman spectroscopy and atomic force microscopy are used for characterization.
    The device shows polarization-dependent second-harmonic generation.
    However, the performance is limited by transfer contamination.
    """

    doc_b = """
    This paper investigates a semiconductor oxide thin film device.
    The film is fabricated by sputtering and post-annealing.
    X-ray diffraction and scanning electron microscopy are used for characterization.
    The results show improved stability after annealing.
    Future work should optimize the processing temperature and film thickness.
    """

    result = compare_documents(
        [doc_a, doc_b],
        document_names=["paper_a", "paper_b"],
        max_snippets_per_dimension=2,
    )

    assert "table" in result
    assert "details" in result
    assert not result["table"].empty
    assert "Dimension" in result["table"].columns
