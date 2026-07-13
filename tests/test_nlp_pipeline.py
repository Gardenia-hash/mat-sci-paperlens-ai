from src.nlp_pipeline import (
    answer_question,
    build_research_brief,
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


def test_research_brief_reports_evidence_coverage_and_complete_sentences():
    brief = build_research_brief(SAMPLE_TEXT)

    assert brief["total_count"] == 5
    assert brief["detected_count"] >= 4
    assert 0.0 <= brief["coverage"] <= 1.0
    detected_items = [item for item in brief["items"] if item["detected"]]
    assert all(str(item["evidence"]).endswith(".") for item in detected_items)
    assert {item["key"] for item in brief["items"]} == {
        "main_focus",
        "materials_system",
        "method",
        "result",
        "limitation",
    }


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


def test_retrieval_does_not_present_zero_similarity_passages_as_evidence():
    results = retrieve_passages(
        ["Raman spectroscopy characterizes the ferroelectric film."],
        query="unrelated battery electrolyte capacity",
        top_k=4,
    )

    assert results == []


def test_page_aware_retrieval_and_answers_preserve_pdf_provenance():
    pages = [
        "The introduction describes a general semiconductor processing challenge.",
        "Atomic layer deposition is used to grow the oxide film at 200 °C.",
        "The conclusion reports improved electrical stability after annealing.",
    ]
    document = "\n\n".join(pages)

    retrieved = retrieve_passages(
        [document],
        query="How is the oxide film grown?",
        top_k=2,
        document_pages=[pages],
    )
    result = answer_question(
        [document],
        query="How is the oxide film grown?",
        document_names=["paper.pdf"],
        document_pages=[pages],
        top_k=2,
    )

    assert retrieved[0]["page_number"] == 2
    assert "[p. 2]" in result["answer"]
    assert "paper.pdf, p. 2" in result["answer"]
    assert result["evidence_strength"] in {"Strong", "Moderate", "Weak"}
    assert result["strength_reason"]


def test_find_domain_hints_has_categories():
    hints = find_domain_hints(SAMPLE_TEXT)
    assert "materials_system" in hints
    assert "characterization" in hints


def test_short_scientific_acronyms_require_term_boundaries():
    hints = find_domain_hints(
        "The so-called baseline procedure remains available for later evaluation."
    )

    assert hints["fabrication_or_synthesis"] == []


def test_summary_and_brief_include_physical_pdf_page_citations():
    pages = [
        "This work studies a ferroelectric thin film for optical switching.",
        "Raman spectroscopy is used to characterize the transferred film.",
        "The results show improved switching stability after annealing.",
    ]
    text = "\n\n".join(pages)

    summary = summarize_text(text, max_sentences=3, page_texts=pages)
    brief = build_research_brief(text, page_texts=pages)

    assert "[p. 1]" in summary
    result_item = next(item for item in brief["items"] if item["key"] == "result")
    assert result_item["page_number"] == 3

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
    assert "[E1]" in result["answer"]
    assert result["answer_items"]


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


def test_comparison_uses_strict_acronym_boundaries_and_page_citations():
    baseline = "The so-called baseline procedure remains available for evaluation."
    deposition = "The oxide film is deposited by ALD under vacuum."

    result = compare_documents(
        [baseline, deposition],
        document_names=["baseline.pdf", "deposition.pdf"],
        document_pages=[[baseline], [deposition]],
    )
    fabrication = result["details"]["Fabrication / synthesis"]

    assert fabrication["baseline.pdf"] == []
    assert fabrication["deposition.pdf"] == [f"{deposition} [p. 1]"]
