import re
from typing import List


def clean_text(text: str) -> str:
    """Normalize PDF-extracted text and remove common scientific-PDF artifacts."""
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove repeated replacement characters and common broken glyphs.
    text = text.replace("�", "")
    text = text.replace("■", "")
    text = text.replace("□", "")
    text = text.replace("▲", "")
    text = text.replace("�", "")

    # Normalize strange spacing around punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text.strip()


def is_noisy_sentence(sentence: str) -> bool:
    """Detect formula-heavy, metadata-heavy, or PDF-artifact-heavy sentences."""
    if not sentence:
        return True

    words = sentence.split()

    if len(words) < 7:
        return True

    # Skip author / affiliation / arXiv metadata lines.
    metadata_patterns = [
        "arxiv:",
        "dated:",
        "university",
        "institute",
        "department",
        "copyright",
        "all rights reserved",
        "corresponding author",
    ]

    lower_sentence = sentence.lower()
    if any(pattern in lower_sentence for pattern in metadata_patterns):
        return True

    # Skip formula-heavy sentences.
    math_symbols = sum(
        sentence.count(symbol)
        for symbol in [
            "≈", "σ", "∆", "Δ", "⊥", "∥", "∞", "∂", "∑", "∫",
            "α", "β", "γ", "θ", "λ", "μ", "π", "ω", "χ",
            "=", "+", "-", "/", "\\", "<", ">", "|",
        ]
    )

    if math_symbols > 8:
        return True

    # Skip lines with too many numbers, equation references, or broken tokens.
    number_like = len(re.findall(r"\b\d+(\.\d+)?\b", sentence))
    if number_like > 8:
        return True

    # Skip sentences with too many single-character tokens.
    single_char_tokens = sum(1 for word in words if len(word) == 1)
    if single_char_tokens / max(len(words), 1) > 0.25:
        return True

    # Skip sentences where alphabetic content is too low.
    alphabetic_chars = sum(ch.isalpha() for ch in sentence)
    if alphabetic_chars / max(len(sentence), 1) < 0.45:
        return True

    return False


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter for English scientific text."""
    text = clean_text(text)
    if not text:
        return []

    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

    sentences = []
    for sentence in candidates:
        sentence = sentence.strip()
        sentence = re.sub(r"\s+", " ", sentence)

        if not is_noisy_sentence(sentence):
            sentences.append(sentence)

    return sentences


def split_passages(text: str, max_words: int = 120) -> List[str]:
    """Split text into paragraph-like passages with a maximum word count."""
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages = []

    for paragraph in paragraphs:
        # Remove extremely noisy paragraphs before retrieval.
        clean_sentences = [
            sentence for sentence in split_sentences(paragraph)
            if not is_noisy_sentence(sentence)
        ]

        if not clean_sentences:
            continue

        paragraph = " ".join(clean_sentences)
        words = paragraph.split()

        if len(words) <= max_words:
            passages.append(paragraph)
        else:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                if len(chunk.split()) >= 20:
                    passages.append(chunk)

    if not passages:
        passages = split_sentences(text)

    return passages
