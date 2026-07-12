from src.text_utils import is_complete_sentence, split_passages, split_sentences


def test_split_sentences_repairs_pdf_hard_wraps():
    text = """
    Abstract
    We demonstrate a stable ferroelectric device
    using a scalable dry-transfer process
    that improves switching uniformity across the sample.
    The measured optical response remains stable
    after repeated electrical cycling under ambient conditions.
    """

    sentences = split_sentences(text)

    assert sentences == [
        "We demonstrate a stable ferroelectric device using a scalable dry-transfer process that improves switching uniformity across the sample.",
        "The measured optical response remains stable after repeated electrical cycling under ambient conditions.",
    ]


def test_split_sentences_protects_scientific_abbreviations():
    text = (
        "The complete measurement sequence is summarized in Figure 1 for reference. "
        "As shown in Fig. 2, the normalized response increases after annealing."
    )

    sentences = split_sentences(text)

    assert len(sentences) == 2
    assert sentences[1].startswith("As shown in Fig. 2")


def test_split_sentences_supports_complete_chinese_sentences():
    text = "本研究制备了具有稳定极化响应的二维铁电器件。结果表明退火处理能够改善器件的循环稳定性。"

    sentences = split_sentences(text)

    assert sentences == [
        "本研究制备了具有稳定极化响应的二维铁电器件。",
        "结果表明退火处理能够改善器件的循环稳定性。",
    ]


def test_incomplete_trailing_fragment_is_not_returned():
    text = (
        "The fabricated device maintains a stable response after repeated cycling. "
        "This trailing extraction fragment contains several words but has no ending"
    )

    sentences = split_sentences(text)

    assert sentences == [
        "The fabricated device maintains a stable response after repeated cycling."
    ]
    assert all(is_complete_sentence(sentence) for sentence in sentences)


def test_short_but_complete_scientific_sentence_is_retained():
    sentences = split_sentences("Mobility increased after annealing.")

    assert sentences == ["Mobility increased after annealing."]


def test_passages_respect_sentence_boundaries_at_word_limit():
    text = (
        "The first complete sentence contains enough words for retrieval testing. "
        "The second complete sentence also contains enough words for reliable testing."
    )

    passages = split_passages(text, max_words=12)

    assert len(passages) == 2
    assert all(is_complete_sentence(passage) for passage in passages)
