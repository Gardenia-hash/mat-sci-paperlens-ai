import re
from typing import List


MOJIBAKE_CHARS = "�□■▲▶◀◆◇●○¤¼½¾ðþÿ�"


def clean_text(text: str) -> str:
    """Normalize PDF-extracted text and remove common scientific-PDF artifacts."""
    if not text:
        return ""

    text = text.replace("\r", "\n")

    # Remove PDF broken hyphenation.
    text = re.sub(r"-\n", "", text)

    # Remove common PDF extraction artifacts.
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"[{}]+".format(re.escape(MOJIBAKE_CHARS)), " ", text)

    # Normalize common mojibake fragments seen in formula-heavy PDFs.
    text = text.replace("¼", " ")
    text = text.replace("ð", " ")
    text = text.replace("¤", " ")
    text = text.replace("ÿ", " ")
    text = text.replace("þ", " ")

    # Remove isolated replacement boxes.
    text = text.replace("�", " ")

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize strange spacing around punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text.strip()


def is_noisy_sentence(sentence: str) -> bool:
    """Detect formula-heavy, metadata-heavy, or PDF-artifact-heavy sentences."""
    if not sentence:
        return True

    sentence = sentence.strip()
    words = sentence.split()
    lower_sentence = sentence.lower()

    if len(words) < 7:
        return True

    if len(words) > 90:
        return True

    metadata_patterns = [
        "arxiv:",
        "dated:",
        "university",
        "institute",
        "department",
        "copyright",
        "all rights reserved",
        "corresponding author",
        "journal of",
        "published by",
        "downloaded from",
        "accepted manuscript",
        "preprint",
    ]

    if any(pattern in lower_sentence for pattern in metadata_patterns):
        return True

    noisy_markers = [
        "fig.",
        "figs.",
        "figure",
        "eq.",
        "equation",
        "appendix",
        "supplementary",
        "table",
    ]

    # One mention of figure can be acceptable, but many formula/figure references are usually bad.
    if sum(marker in lower_sentence for marker in noisy_markers) >= 2:
        return True

    weird_char_count = sum(sentence.count(ch) for ch in MOJIBAKE_CHARS)
    if weird_char_count > 1:
        return True

    math_symbols = sum(
        sentence.count(symbol)
        for symbol in [
            "≈", "σ", "∆", "Δ", "⊥", "∥", "∞", "∂", "∑", "∫",
            "α", "β", "γ", "θ", "λ", "μ", "π", "ω", "χ", "η",
            "=", "+", "/", "\\", "<", ">", "|", "±", "√", "∇",
        ]
    )

    if math_symbols > 6:
        return True

    # Too many numbers usually means equation, table, page header, or reference noise.
    number_like = len(re.findall(r"\b\d+(\.\d+)?\b", sentence))
    if number_like > 8:
        return True

    # Too many single-character tokens usually means broken formula text.
    single_char_tokens = sum(1 for word in words if len(word) == 1)
    if single_char_tokens / max(len(words), 1) > 0.22:
        return True

    alphabetic_chars = sum(ch.isalpha() for ch in sentence)
    if alphabetic_chars / max(len(sentence), 1) < 0.48:
        return True

    # Filter obvious page/footer lines.
    if re.search(r"^\d+\s*$", sentence):
        return True

    if re.search(r"^\d{5,}-\d+", sentence):
        return True

    return False


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter for English scientific text."""
    text = clean_text(text)
    if not text:
        return []

    # Split both paragraph lines and normal sentence boundaries.
    raw_candidates = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        raw_candidates.extend(parts)

    sentences = []
    for sentence in raw_candidates:
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
