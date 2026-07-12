from src.nlp_pipeline import (
    answer_question,
    compare_documents,
    extract_keywords,
    infer_question_type,
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


def test_summary_reconstructs_wrapped_lines_into_integrated_overview():
    wrapped_text = """
    Abstract
    We demonstrate a stable ferroelectric device
    using a scalable dry-transfer process
    that improves switching uniformity across the sample.
    Raman spectroscopy and atomic force microscopy
    are used to characterize the transferred layers.
    The device maintains a stable optical response
    after repeated electrical cycling under ambient conditions.
    """

    summary = summarize_text(wrapped_text, max_sentences=5)

    assert "### Grounded summary" in summary
    assert "device using a scalable dry-transfer process that improves" in summary
    assert "microscopy are used to characterize" in summary


def test_extract_keywords_returns_dataframe():
    keywords = extract_keywords(SAMPLE_TEXT, top_n=5)
    assert list(keywords.columns) == ["keyword", "score"]
    assert len(keywords) > 0


def test_retrieve_passages_returns_relevant_result():
    results = retrieve_passages([SAMPLE_TEXT], query="What characterization methods are used?", top_k=2)
    assert len(results) > 0
    assert "passage" in results[0]
    assert "passage_index" in results[0]
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


def test_question_intent_uses_whole_words_instead_of_substrings():
    assert infer_question_type("What does the result show?") == "result"
    assert infer_question_type("How is the sample prepared?") == "method"


def test_answer_uses_complete_reconstructed_sentences():
    wrapped_text = """
    The sample is characterized by Raman spectroscopy
    and atomic force microscopy after dry transfer.
    The measurements confirm uniform thickness across
    the active region of the fabricated device.
    """

    result = answer_question(
        [wrapped_text],
        query="What characterization methods are used?",
        document_names=["paper.pdf"],
        top_k=2,
    )

    assert "Raman spectroscopy and atomic force microscopy after dry transfer." in result["answer"]
    assert "**Answer:**\n- " not in result["answer"]


def test_answer_integrates_selected_evidence_in_document_order():
    text = (
        "The sample is first prepared by dry transfer onto a silicon substrate. "
        "Raman spectroscopy is then used to verify the transferred material quality. "
        "Atomic force microscopy subsequently confirms a uniform surface morphology."
    )

    result = answer_question(
        [text],
        query="How is the sample prepared and characterized by Raman and microscopy?",
        document_names=["paper.pdf"],
        top_k=3,
        max_answer_sentences=3,
    )

    answer = result["answer"]
    assert result["question_type"] == "method"
    assert answer.index("first prepared") < answer.index("Raman spectroscopy")
    assert answer.index("Raman spectroscopy") < answer.index("Atomic force microscopy")


def test_answer_excludes_weakly_related_sentences_from_integrated_output():
    text = (
        "Raman spectroscopy and atomic force microscopy are used to characterize the transferred layers. "
        "The optical response remains stable after repeated cycling under ambient conditions."
    )

    result = answer_question(
        [text],
        query="Which characterization methods are used?",
        document_names=["paper.pdf"],
        top_k=2,
        max_answer_sentences=4,
    )

    assert "Raman spectroscopy and atomic force microscopy" in result["answer"]
    assert "optical response remains stable" not in result["answer"]


def test_multi_document_answer_keeps_integrated_outputs_separate_by_source():
    result = answer_question(
        [
            "Raman spectroscopy characterizes the transferred ferroelectric layer.",
            "X-ray diffraction characterizes the annealed semiconductor film.",
        ],
        query="Which characterization methods are used?",
        document_names=["paper_a.pdf", "paper_b.pdf"],
        top_k=4,
        max_answer_sentences=4,
    )

    answer = result["answer"]
    assert "**paper_a.pdf:**" in answer
    assert "**paper_b.pdf:**" in answer

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
