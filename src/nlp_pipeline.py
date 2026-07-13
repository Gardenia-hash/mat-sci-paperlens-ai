from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_utils import clean_text, is_complete_sentence, split_passages, split_sentences


SECTION_ALIASES = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "methods": [
        "method",
        "methods",
        "methodology",
        "experimental",
        "experiment",
        "materials and methods",
        "model",
        "simulation",
        "computational methods",
    ],
    "results": ["result", "results", "findings"],
    "discussion": ["discussion"],
    "conclusion": [
        "conclusion",
        "conclusions",
        "summary",
        "outlook",
        "future work",
    ],
    "references": ["references", "bibliography"],
}


SUMMARY_ROLE_KEYWORDS = {
    "main_focus": [
        "we study",
        "we investigate",
        "we report",
        "we present",
        "we propose",
        "this work",
        "this paper",
        "aim",
        "focus",
    ],
    "materials_system": [
        "material",
        "materials",
        "sample",
        "compound",
        "thin film",
        "heterostructure",
        "semiconductor",
        "ferroelectric",
        "oxide",
        "chalcogenide",
        "wafer",
        "device",
    ],
    "method": [
        "prepared",
        "fabricated",
        "synthesized",
        "deposited",
        "measured",
        "characterized",
        "simulated",
        "calculated",
        "using",
        "method",
        "model",
        "experiment",
    ],
    "result": [
        "show",
        "shows",
        "demonstrate",
        "demonstrates",
        "found",
        "observed",
        "revealed",
        "result",
        "increase",
        "decrease",
        "improvement",
        "enhanced",
        "suppressed",
    ],
    "limitation": [
        "however",
        "limitation",
        "limited",
        "challenge",
        "future work",
        "requires",
        "uncertain",
        "remain",
        "further",
    ],
}


DOMAIN_KEYWORDS = {
    "materials_system": [
        "film",
        "thin film",
        "heterostructure",
        "ferroelectric",
        "semiconductor",
        "skyrmion",
        "vdw",
        "van der waals",
        "oxide",
        "chalcogenide",
        "silicon",
        "wafer",
    ],
    "fabrication_or_synthesis": [
        "synthesis",
        "growth",
        "deposition",
        "annealing",
        "lithography",
        "etching",
        "sputtering",
        "cvd",
        "mocvd",
        "ald",
        "exfoliation",
    ],
    "characterization": [
        "sem",
        "tem",
        "fib",
        "xrd",
        "x-ray diffraction",
        "x ray diffraction",
        "raman",
        "afm",
        "ebsd",
        "shg",
        "ellipsometry",
        "metrology",
        "microscopy",
        "spectroscopy",
        "characteriz",
        "diffract",
    ],
    "results_or_performance": [
        "increase",
        "decrease",
        "improvement",
        "mobility",
        "coercive",
        "polarization",
        "switching",
        "stability",
        "efficiency",
        "accuracy",
        "defect density",
    ],
    "limitations_or_future_work": [
        "limitation",
        "challenge",
        "future work",
        "however",
        "requires",
        "uncertain",
        "noise",
        "artifact",
        "scalability",
    ],
}


def _normalize_heading(line: str) -> str:
    """Normalize possible section headings from PDF text."""
    line = line.strip().lower()
    line = re.sub(r"^\s*(\d+|[ivxlcdm]+)\.?\s+", "", line)
    line = re.sub(r"[^a-z\s]", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _detect_section_heading(line: str) -> str | None:
    """Detect common scientific paper section headings."""
    if not line or len(line.split()) > 8:
        return None

    normalized = _normalize_heading(line)

    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if normalized == alias or normalized.startswith(alias + " "):
                return section

    return None


def extract_sections(text: str) -> Dict[str, str]:
    """Split a scientific paper into rough sections based on headings."""
    text = clean_text(text)
    sections: Dict[str, List[str]] = {"body": []}
    current_section = "body"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        detected = _detect_section_heading(line)
        if detected:
            current_section = detected
            if current_section not in sections:
                sections[current_section] = []
            continue

        if current_section == "references":
            continue

        sections.setdefault(current_section, []).append(line)

    return {
        section: "\n".join(lines)
        for section, lines in sections.items()
        if lines
    }


def _clean_sentence_for_summary(sentence: str) -> str:
    """Light cleanup without changing the scientific meaning."""
    sentence = sentence.strip()

    # Remove citation marks like [1], [2,3], [4-6].
    sentence = re.sub(r"\[[\d,\-\s]+\]", "", sentence)

    # Remove repeated spaces.
    sentence = re.sub(r"\s+", " ", sentence)

    # Remove spaces before punctuation.
    sentence = re.sub(r"\s+([,.;:!?])", r"\1", sentence)

    return sentence.strip()


def _is_bad_summary_sentence(sentence: str) -> bool:
    """Filter sentences unsuitable for a readable paper summary."""
    if not sentence:
        return True

    if not is_complete_sentence(sentence):
        return True

    words = sentence.split()
    lower_sentence = sentence.lower()
    cjk_char_count = len(re.findall(r"[\u3400-\u9fff]", sentence))
    has_cjk_text = cjk_char_count >= 6

    if has_cjk_text and cjk_char_count < 8:
        return True

    if not has_cjk_text and len(words) < 4:
        return True

    if (has_cjk_text and len(sentence) > 450) or (not has_cjk_text and len(words) > 85):
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
        "email:",
    ]

    if any(pattern in lower_sentence for pattern in metadata_patterns):
        return True

    math_symbols = sum(
        sentence.count(symbol)
        for symbol in [
            "≈",
            "σ",
            "∆",
            "Δ",
            "⊥",
            "∥",
            "∞",
            "∂",
            "∑",
            "∫",
            "α",
            "β",
            "γ",
            "θ",
            "λ",
            "μ",
            "π",
            "ω",
            "χ",
            "=",
            "+",
            "/",
            "\\",
            "<",
            ">",
            "|",
        ]
    )

    if math_symbols > 7:
        return True

    number_like = len(re.findall(r"\b\d+(\.\d+)?\b", sentence))
    if number_like > 8:
        return True

    single_char_tokens = sum(1 for word in words if len(word) == 1)
    if not has_cjk_text and single_char_tokens / max(len(words), 1) > 0.22:
        return True

    alphabetic_chars = sum(ch.isalpha() for ch in sentence)
    if alphabetic_chars / max(len(sentence), 1) < 0.45:
        return True

    noisy_phrases = [
        "eq.",
        "equation",
        "fig.",
        "figure",
        "table",
        "appendix",
        "supplementary",
    ]

    if sum(phrase in lower_sentence for phrase in noisy_phrases) >= 2:
        return True

    return False


def _section_weight(section: str) -> float:
    """Give higher priority to sections that usually contain summary-worthy content."""
    weights = {
        "abstract": 1.45,
        "conclusion": 1.35,
        "results": 1.22,
        "discussion": 1.15,
        "methods": 1.05,
        "introduction": 1.00,
        "body": 0.95,
    }
    return weights.get(section, 1.0)


def _role_bonus(sentence: str) -> float:
    """Reward sentences that look like main claims, methods, results, or limitations."""
    lower_sentence = sentence.lower()
    bonus = 0.0

    for role_keywords in SUMMARY_ROLE_KEYWORDS.values():
        if any(keyword in lower_sentence for keyword in role_keywords):
            bonus += 0.25

    domain_terms = []
    for terms in DOMAIN_KEYWORDS.values():
        domain_terms.extend(terms)

    domain_hits = sum(1 for term in domain_terms if term in lower_sentence)
    bonus += min(domain_hits * 0.06, 0.36)

    return bonus


def _build_summary_candidates(text: str) -> List[Dict[str, object]]:
    """Create cleaned and section-aware summary candidates."""
    sections = extract_sections(text)
    candidates: List[Dict[str, object]] = []

    for section, section_text in sections.items():
        sentences = split_sentences(section_text)

        for position, sentence in enumerate(sentences):
            sentence = _clean_sentence_for_summary(sentence)

            if _is_bad_summary_sentence(sentence):
                continue

            candidates.append(
                {
                    "section": section,
                    "position": position,
                    "order": len(candidates),
                    "sentence": sentence,
                }
            )

    return candidates


def _select_non_redundant_sentences(
    candidates: List[Dict[str, object]],
    scores,
    matrix,
    max_sentences: int,
) -> List[Dict[str, object]]:
    """Select high-score sentences while avoiding repetitive sentences."""
    ranked_indices = scores.argsort()[::-1]
    selected_indices: List[int] = []

    for index in ranked_indices:
        if len(selected_indices) >= max_sentences:
            break

        if not selected_indices:
            selected_indices.append(index)
            continue

        is_redundant = False
        for selected_index in selected_indices:
            similarity = cosine_similarity(matrix[index], matrix[selected_index])[0][0]
            if similarity > 0.68:
                is_redundant = True
                break

        if not is_redundant:
            selected_indices.append(index)

    selected = [candidates[index] for index in selected_indices]
    return selected


def _find_best_by_role(
    candidates: List[Dict[str, object]],
    selected: List[Dict[str, object]],
    role: str,
) -> str | None:
    """Find the best sentence for a summary role."""
    role_terms = SUMMARY_ROLE_KEYWORDS[role]

    preferred_sections = {
        "main_focus": ["abstract", "introduction", "body"],
        "materials_system": ["abstract", "methods", "introduction", "body"],
        "method": ["methods", "abstract", "body"],
        "result": ["results", "discussion", "conclusion", "abstract", "body"],
        "limitation": ["conclusion", "discussion", "results", "body"],
    }

    pool = selected + candidates

    for section in preferred_sections[role]:
        for item in pool:
            sentence = str(item["sentence"])
            lower_sentence = sentence.lower()
            if item["section"] == section and any(term in lower_sentence for term in role_terms):
                return sentence

    for item in pool:
        sentence = str(item["sentence"])
        lower_sentence = sentence.lower()
        if any(term in lower_sentence for term in role_terms):
            return sentence

    return None


def build_research_brief(text: str) -> Dict[str, object]:
    """Build a five-part, evidence-backed paper brief for first-pass reading.

    Each detected item is a complete sentence copied from the source text.  The
    coverage score describes extraction coverage only; it is not a judgement of
    paper quality or scientific certainty.
    """
    candidates = _build_summary_candidates(text)
    brief_roles = (
        "main_focus",
        "materials_system",
        "method",
        "result",
        "limitation",
    )
    items = []

    for role in brief_roles:
        evidence = _find_best_by_role(candidates, [], role)
        section = None
        if evidence:
            matching_item = next(
                (item for item in candidates if item["sentence"] == evidence),
                None,
            )
            if matching_item:
                section = str(matching_item["section"])

        items.append(
            {
                "key": role,
                "evidence": evidence,
                "section": section,
                "detected": evidence is not None,
            }
        )

    detected_count = sum(bool(item["detected"]) for item in items)
    return {
        "items": items,
        "detected_count": detected_count,
        "total_count": len(items),
        "coverage": detected_count / len(items) if items else 0.0,
    }


def summarize_text(text: str, max_sentences: int = 5) -> str:
    """Create a grounded, section-aware, readable extractive summary."""
    candidates = _build_summary_candidates(text)

    if not candidates:
        return (
            "No clean readable sentence was found. "
            "The uploaded PDF may contain mostly equations, scanned text, or broken encoding."
        )

    candidate_sentences = [str(item["sentence"]) for item in candidates]

    if len(candidate_sentences) <= max_sentences:
        selected = candidates
    else:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.90,
                min_df=1,
            )
            matrix = vectorizer.fit_transform(candidate_sentences)
            base_scores = matrix.sum(axis=1).A1
        except ValueError:
            selected = candidates[:max_sentences]
        else:
            final_scores = base_scores.copy()

            for index, item in enumerate(candidates):
                section = str(item["section"])
                position = int(item["position"])
                sentence = str(item["sentence"])

                final_scores[index] *= _section_weight(section)

                if position <= 2:
                    final_scores[index] *= 1.10

                final_scores[index] += _role_bonus(sentence)

            selected = _select_non_redundant_sentences(
                candidates=candidates,
                scores=final_scores,
                matrix=matrix,
                max_sentences=max_sentences,
            )

    main_focus = _find_best_by_role(candidates, selected, "main_focus")
    method = _find_best_by_role(candidates, selected, "method")
    result = _find_best_by_role(candidates, selected, "result")
    limitation = _find_best_by_role(candidates, selected, "limitation")

    ordered_selected = sorted(selected, key=lambda item: int(item.get("order", 0)))
    integrated_overview = " ".join(str(item["sentence"]) for item in ordered_selected)
    integrated_sentences = {str(item["sentence"]) for item in ordered_selected}

    used_sentences = set()
    lines = ["### Grounded summary", "", integrated_overview]

    role_lines = []

    if main_focus and main_focus not in integrated_sentences:
        role_lines.append(f"- **Main focus:** {main_focus}")
        used_sentences.add(main_focus)

    if method and method not in used_sentences and method not in integrated_sentences:
        role_lines.append(f"- **Approach / method:** {method}")
        used_sentences.add(method)

    if result and result not in used_sentences and result not in integrated_sentences:
        role_lines.append(f"- **Key result:** {result}")
        used_sentences.add(result)

    if limitation and limitation not in used_sentences and limitation not in integrated_sentences:
        role_lines.append(f"- **Limitation / future work:** {limitation}")
        used_sentences.add(limitation)

    if role_lines:
        lines.append("")
        lines.append("### Supporting points")
        lines.extend(role_lines)

    lines.append("")
    lines.append(
        "> Note: This is a grounded extractive summary. The wording is lightly cleaned, "
        "but the factual content is selected from the original document."
    )

    return "\n".join(lines)


def extract_keywords(text: str, top_n: int = 15) -> pd.DataFrame:
    """Extract top TF-IDF keywords and short phrases from one document."""
    text = clean_text(text)

    if not text:
        return pd.DataFrame(columns=["keyword", "score"])

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500,
        )
        matrix = vectorizer.fit_transform([text])
    except ValueError:
        return pd.DataFrame(columns=["keyword", "score"])

    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]

    ranked = sorted(
        zip(feature_names, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    return pd.DataFrame(ranked, columns=["keyword", "score"])


def retrieve_passages(
    documents: List[str],
    query: str,
    top_k: int = 4,
) -> List[Dict[str, object]]:
    """Retrieve passages that are most similar to the query."""
    all_passages = []
    passage_to_doc = []
    passage_positions = []

    for doc_index, doc in enumerate(documents):
        for passage_index, passage in enumerate(split_passages(doc)):
            all_passages.append(passage)
            passage_to_doc.append(doc_index)
            passage_positions.append(passage_index)

    if not all_passages or not query.strip():
        return []

    corpus = all_passages + [query]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return []

    query_vector = matrix[-1]
    passage_matrix = matrix[:-1]
    similarities = cosine_similarity(query_vector, passage_matrix).ravel()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for index in top_indices:
        results.append(
            {
                "document_index": passage_to_doc[index],
                "passage_index": passage_positions[index],
                "passage": all_passages[index],
                "score": float(similarities[index]),
            }
        )

    return results

QUESTION_TYPE_PATTERNS = {
    "materials_system": [
        "material",
        "materials",
        "system",
        "sample",
        "compound",
        "film",
        "structure",
        "what is studied",
        "材料",
        "体系",
        "样品",
        "研究对象",
        "研究了什么",
    ],
    "method": [
        "method",
        "methods",
        "approach",
        "prepared",
        "fabricated",
        "synthesized",
        "characterized",
        "measured",
        "simulated",
        "calculated",
        "how",
        "方法",
        "怎么做",
        "如何",
        "制备",
        "表征",
        "测量",
        "模拟",
        "计算",
    ],
    "result": [
        "result",
        "results",
        "finding",
        "findings",
        "performance",
        "show",
        "shows",
        "demonstrate",
        "conclusion",
        "结果",
        "发现",
        "性能",
        "说明",
        "证明",
        "结论",
    ],
    "limitation": [
        "limitation",
        "limitations",
        "challenge",
        "challenges",
        "problem",
        "future work",
        "drawback",
        "weakness",
        "限制",
        "局限",
        "问题",
        "挑战",
        "不足",
        "未来",
    ],
    "parameter": [
        "parameter",
        "parameters",
        "temperature",
        "field",
        "current",
        "voltage",
        "thickness",
        "size",
        "dimension",
        "frequency",
        "value",
        "参数",
        "设定",
        "温度",
        "电流",
        "电压",
        "厚度",
        "尺寸",
        "数值",
        "磁场",
    ],
}


QUERY_EXPANSION_TERMS = {
    "materials_system": [
        "material",
        "sample",
        "compound",
        "thin film",
        "heterostructure",
        "semiconductor",
        "ferroelectric",
        "oxide",
        "chalcogenide",
        "wafer",
    ],
    "method": [
        "method",
        "fabrication",
        "synthesis",
        "deposition",
        "annealing",
        "lithography",
        "etching",
        "characterization",
        "measurement",
        "simulation",
        "calculation",
        "SEM",
        "TEM",
        "XRD",
        "Raman",
        "AFM",
    ],
    "result": [
        "result",
        "show",
        "demonstrate",
        "observe",
        "increase",
        "decrease",
        "improve",
        "enhance",
        "performance",
        "stability",
        "efficiency",
        "switching",
    ],
    "limitation": [
        "limitation",
        "challenge",
        "however",
        "future work",
        "requires",
        "uncertain",
        "noise",
        "artifact",
        "scalability",
    ],
    "parameter": [
        "parameter",
        "temperature",
        "current",
        "voltage",
        "field",
        "thickness",
        "size",
        "dimension",
        "frequency",
        "value",
        "rate",
        "time",
    ],
    "general": [
        "study",
        "paper",
        "work",
        "method",
        "result",
        "material",
        "performance",
    ],
}


ANSWER_BONUS_TERMS = {
    "materials_system": DOMAIN_KEYWORDS["materials_system"],
    "method": DOMAIN_KEYWORDS["fabrication_or_synthesis"] + DOMAIN_KEYWORDS["characterization"],
    "result": DOMAIN_KEYWORDS["results_or_performance"],
    "limitation": DOMAIN_KEYWORDS["limitations_or_future_work"],
    "parameter": [
        "temperature",
        "current",
        "voltage",
        "field",
        "thickness",
        "size",
        "dimension",
        "frequency",
        "parameter",
        "value",
        "rate",
        "time",
    ],
    "general": [],
}


def infer_question_type(query: str) -> str:
    """Infer the user's question type from English or Chinese keywords."""
    lower_query = query.lower()
    scores: Dict[str, float] = {}

    for question_type, patterns in QUESTION_TYPE_PATTERNS.items():
        if question_type == "general":
            continue

        score = 0.0
        for pattern in patterns:
            normalized_pattern = pattern.lower()
            if re.search(r"[\u3400-\u9fff]", normalized_pattern):
                pattern_found = normalized_pattern in lower_query
            else:
                boundary_pattern = (
                    r"(?<!\w)"
                    + re.escape(normalized_pattern).replace(r"\ ", r"\s+")
                    + r"(?!\w)"
                )
                pattern_found = bool(re.search(boundary_pattern, lower_query))

            if pattern_found:
                token_count = len(normalized_pattern.split())
                score += 1.0 + min(token_count - 1, 2) * 0.35

        scores[question_type] = score

    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score > 0 else "general"


def expand_query(query: str, question_type: str) -> str:
    """Expand a user question with English domain terms.

    This helps Chinese questions retrieve English paper passages.
    """
    expansion_terms = QUERY_EXPANSION_TERMS.get(question_type, QUERY_EXPANSION_TERMS["general"])
    return query + " " + " ".join(expansion_terms)


def answer_sentence_bonus(sentence: str, question_type: str) -> float:
    """Reward sentences that match the detected question intent."""
    lower_sentence = sentence.lower()
    bonus_terms = ANSWER_BONUS_TERMS.get(question_type, [])

    bonus = 0.0
    for term in bonus_terms:
        if term.lower() in lower_sentence:
            bonus += 0.08

    # Sentences with direct claim language are often useful answers.
    claim_terms = [
        "we show",
        "we demonstrate",
        "we find",
        "we observe",
        "the results",
        "this suggests",
        "indicating that",
        "therefore",
        "however",
    ]

    if any(term in lower_sentence for term in claim_terms):
        bonus += 0.15

    return min(bonus, 0.5)


def deduplicate_answer_sentences(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Remove repeated or near-identical answer sentences."""
    seen = set()
    unique_items = []

    for item in items:
        sentence = str(item["sentence"])
        normalized = re.sub(r"[^a-zA-Z0-9]+", " ", sentence.lower()).strip()

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_items.append(item)

    return unique_items


def answer_question(
    documents: List[str],
    query: str,
    document_names: List[str] | None = None,
    top_k: int = 4,
    max_answer_sentences: int = 4,
) -> Dict[str, object]:
    """Generate a grounded answer from retrieved paper passages.

    The answer is extractive and evidence-based. It does not invent facts
    beyond the uploaded document text.
    """
    if not query.strip():
        return {
            "answer": "Please enter a question.",
            "evidence": [],
            "answer_items": [],
            "question_type": "general",
            "confidence": "Low",
        }

    question_type = infer_question_type(query)
    expanded_query = expand_query(query, question_type)

    retrieved = retrieve_passages(
        documents=documents,
        query=expanded_query,
        top_k=top_k,
    )

    if not retrieved:
        return {
            "answer": (
                "### Grounded answer\n\n"
                "I could not find relevant evidence in the uploaded document. "
                "Try asking a more specific question, or upload a cleaner PDF/text file."
            ),
            "evidence": [],
            "answer_items": [],
            "question_type": question_type,
            "confidence": "Low",
        }

    candidate_items: List[Dict[str, object]] = []

    for passage_item in retrieved:
        passage = str(passage_item["passage"])
        doc_index = int(passage_item["document_index"])
        passage_index = int(passage_item.get("passage_index", 0))
        passage_score = float(passage_item["score"])

        source_name = (
            document_names[doc_index]
            if document_names and doc_index < len(document_names)
            else f"Document {doc_index + 1}"
        )

        for sentence_index, sentence in enumerate(split_sentences(passage)):
            sentence = _clean_sentence_for_summary(sentence)

            if _is_bad_summary_sentence(sentence):
                continue

            candidate_items.append(
                {
                    "sentence": sentence,
                    "source": source_name,
                    "document_index": doc_index,
                    "passage_index": passage_index,
                    "sentence_index": sentence_index,
                    "passage_score": passage_score,
                }
            )

    if not candidate_items:
        return {
            "answer": (
                "### Grounded answer\n\n"
                "I found potentially relevant passages, but they were too noisy "
                "to form a clean answer. The PDF may contain dense formulas, broken text, or scanned content."
            ),
            "evidence": retrieved,
            "answer_items": [],
            "question_type": question_type,
            "confidence": "Low",
        }

    candidate_sentences = [str(item["sentence"]) for item in candidate_items]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(candidate_sentences + [expanded_query])
        query_vector = matrix[-1]
        sentence_matrix = matrix[:-1]
        similarities = cosine_similarity(query_vector, sentence_matrix).ravel()
    except ValueError:
        similarities = [0.0] * len(candidate_items)

    scored_items = []

    for index, item in enumerate(candidate_items):
        sentence = str(item["sentence"])
        sentence_score = float(similarities[index])
        passage_score = float(item["passage_score"])
        bonus = answer_sentence_bonus(sentence, question_type)

        final_score = sentence_score + 0.35 * passage_score + bonus

        scored_item = dict(item)
        scored_item["answer_score"] = final_score
        scored_items.append(scored_item)

    scored_items = sorted(
        scored_items,
        key=lambda item: float(item["answer_score"]),
        reverse=True,
    )

    unique_items = deduplicate_answer_sentences(scored_items)
    best_available_score = float(unique_items[0]["answer_score"]) if unique_items else 0.0
    minimum_answer_score = max(0.12, best_available_score * 0.42)
    selected_items = [
        item
        for item in unique_items
        if float(item["answer_score"]) >= minimum_answer_score
    ][:max_answer_sentences]

    if not selected_items and unique_items:
        selected_items = unique_items[:1]

    best_score = float(selected_items[0]["answer_score"]) if selected_items else 0.0

    if best_score >= 0.35:
        confidence = "High"
    elif best_score >= 0.18:
        confidence = "Medium"
    else:
        confidence = "Low"

    lines = [
        "### Grounded answer",
        "",
        f"**Detected question type:** `{question_type}`",
        f"**Confidence:** `{confidence}`",
        "",
    ]

    if confidence == "Low":
        lines.append(
            "> The answer below is based on weak textual evidence. "
            "Please verify it against the original paper."
        )
        lines.append("")

    lines.append("**Answer:**")
    evidence_ids = {
        (int(item["document_index"]), int(item.get("passage_index", 0))): index
        for index, item in enumerate(retrieved, start=1)
    }
    answers_by_source: Dict[str, List[str]] = {}
    selected_for_output = sorted(
        selected_items,
        key=lambda item: (
            int(item["document_index"]),
            int(item.get("passage_index", 0)),
            int(item.get("sentence_index", 0)),
        ),
    )
    for item in selected_for_output:
        source = str(item["source"])
        evidence_id = evidence_ids.get(
            (
                int(item["document_index"]),
                int(item.get("passage_index", 0)),
            )
        )
        citation = f" [E{evidence_id}]" if evidence_id else ""
        item["evidence_id"] = evidence_id
        answers_by_source.setdefault(source, []).append(
            f"{item['sentence']}{citation}"
        )

    if len(answers_by_source) == 1:
        lines.append(" ".join(next(iter(answers_by_source.values()))))
    else:
        for source, source_sentences in answers_by_source.items():
            lines.append(f"- **{source}:** {' '.join(source_sentences)}")

    lines.append("")
    lines.append("**Evidence sources:**")
    used_sources = []

    for item in selected_items:
        source = str(item["source"])
        if source not in used_sources:
            used_sources.append(source)

    for source in used_sources:
        lines.append(f"- {source}")

    lines.append("")
    lines.append(
        "> Note: This QA mode is extractive and grounded. "
        "It answers by selecting relevant sentences from the uploaded document, not by inventing new claims."
    )

    return {
        "answer": "\n".join(lines),
        "evidence": retrieved,
        "answer_items": selected_for_output,
        "question_type": question_type,
        "confidence": confidence,
    }

COMPARISON_DIMENSIONS = {
    "Research focus": [
        "we study",
        "we investigate",
        "we report",
        "we present",
        "we propose",
        "this work",
        "this paper",
        "aim",
        "focus",
        "objective",
        "goal",
    ],
    "Material / system": DOMAIN_KEYWORDS["materials_system"]
    + [
        "sample",
        "compound",
        "material system",
        "device structure",
        "layered",
        "hybrid structure",
    ],
    "Fabrication / synthesis": DOMAIN_KEYWORDS["fabrication_or_synthesis"]
    + [
        "prepared",
        "fabricated",
        "transferred",
        "grown",
        "processed",
        "patterned",
    ],
    "Characterization / measurement": DOMAIN_KEYWORDS["characterization"]
    + [
        "measurement",
        "measured",
        "characterized",
        "spectroscopy",
        "diffraction",
        "imaging",
        "electrical measurement",
        "optical measurement",
    ],
    "Parameters / settings": [
        "temperature",
        "current",
        "voltage",
        "field",
        "magnetic field",
        "electric field",
        "thickness",
        "width",
        "length",
        "diameter",
        "size",
        "dimension",
        "frequency",
        "power",
        "pressure",
        "time",
        "rate",
        "dose",
        "concentration",
        "parameter",
        "setting",
        "condition",
    ],
    "Key results / performance": DOMAIN_KEYWORDS["results_or_performance"]
    + [
        "show",
        "shows",
        "demonstrate",
        "demonstrates",
        "observed",
        "revealed",
        "achieved",
        "performance",
        "response",
        "signal",
        "effect",
    ],
    "Limitations / future work": DOMAIN_KEYWORDS["limitations_or_future_work"]
    + [
        "drawback",
        "weakness",
        "future direction",
        "further study",
        "needs",
        "need to",
        "still",
        "remain",
    ],
}


PARAMETER_UNIT_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?"
    r"(nm|μm|um|mm|cm|m|k|K|°C|c|T|mT|Oe|A|mA|uA|μA|V|mV|eV|meV|"
    r"Hz|kHz|MHz|GHz|s|ms|ns|min|h|Pa|bar|rpm|%|wt%|at%)\b"
)


def _sentence_dimension_score(sentence: str, dimension: str, keywords: List[str]) -> float:
    """Score how relevant a sentence is to a comparison dimension."""
    lower_sentence = sentence.lower()
    score = 0.0

    for keyword in keywords:
        if keyword.lower() in lower_sentence:
            score += 1.0

    if dimension == "Parameters / settings":
        unit_hits = PARAMETER_UNIT_PATTERN.findall(sentence)
        score += 1.5 * len(unit_hits)

        # Extra reward for sentences that look like experimental/simulation conditions.
        condition_terms = [
            "under",
            "at",
            "with",
            "using",
            "condition",
            "parameter",
            "set to",
            "fixed",
            "varied",
        ]
        if any(term in lower_sentence for term in condition_terms):
            score += 0.8

    if dimension == "Research focus":
        if len(sentence.split()) <= 35:
            score += 0.3

    if dimension == "Key results / performance":
        result_terms = [
            "show",
            "shows",
            "demonstrate",
            "demonstrates",
            "indicate",
            "suggest",
            "therefore",
            "as a result",
        ]
        if any(term in lower_sentence for term in result_terms):
            score += 0.8

    return score


def _extract_dimension_snippets(
    text: str,
    dimension: str,
    keywords: List[str],
    max_snippets: int = 3,
) -> List[str]:
    """Extract the most relevant evidence sentences for one comparison dimension."""
    sentences = split_sentences(text)
    scored = []

    for sentence_index, sentence in enumerate(sentences):
        sentence = _clean_sentence_for_summary(sentence)

        if _is_bad_summary_sentence(sentence):
            continue

        score = _sentence_dimension_score(sentence, dimension, keywords)

        if score > 0:
            scored.append(
                {
                    "sentence": sentence,
                    "score": score,
                    "position": sentence_index,
                }
            )

    scored = sorted(scored, key=lambda item: item["score"], reverse=True)

    selected = []
    seen = set()

    for item in scored:
        sentence = item["sentence"]
        normalized = re.sub(r"[^a-zA-Z0-9]+", " ", sentence.lower()).strip()

        if normalized in seen:
            continue

        seen.add(normalized)
        selected.append(item)

        if len(selected) >= max_snippets:
            break

    selected = sorted(selected, key=lambda item: int(item["position"]))
    return [str(item["sentence"]) for item in selected]


def _format_comparison_cell(snippets: List[str]) -> str:
    """Format snippets for display in a comparison table."""
    if not snippets:
        return "Not clearly detected."

    return " ".join(snippets)


def _extract_top_keyword_set(text: str, top_n: int = 15) -> set:
    """Extract a set of top keywords from one document."""
    keyword_df = extract_keywords(text, top_n=top_n)

    if keyword_df.empty:
        return set()

    return set(keyword_df["keyword"].astype(str).str.lower().tolist())


def compare_documents(
    documents: List[str],
    document_names: List[str],
    max_snippets_per_dimension: int = 3,
    top_keywords: int = 15,
) -> Dict[str, object]:
    """Compare two or more papers across research-relevant dimensions.

    The comparison is extractive and grounded in the uploaded documents.
    """
    if len(documents) < 2:
        return {
            "table": pd.DataFrame(),
            "details": {},
            "keyword_table": pd.DataFrame(),
            "note": "Please upload at least two documents for comparison.",
        }

    safe_names = [
        name if name else f"Document {index + 1}"
        for index, name in enumerate(document_names)
    ]

    comparison_rows = []
    details: Dict[str, Dict[str, List[str]]] = {}

    for dimension, keywords in COMPARISON_DIMENSIONS.items():
        row = {"Dimension": dimension}
        details[dimension] = {}

        for doc_index, document in enumerate(documents):
            doc_name = safe_names[doc_index]
            snippets = _extract_dimension_snippets(
                text=document,
                dimension=dimension,
                keywords=keywords,
                max_snippets=max_snippets_per_dimension,
            )

            details[dimension][doc_name] = snippets
            row[doc_name] = _format_comparison_cell(snippets)

        comparison_rows.append(row)

    comparison_table = pd.DataFrame(comparison_rows)

    keyword_sets = []
    for doc in documents:
        keyword_sets.append(_extract_top_keyword_set(doc, top_n=top_keywords))

    common_keywords = set.intersection(*keyword_sets) if keyword_sets else set()

    keyword_rows = []

    for doc_index, keyword_set in enumerate(keyword_sets):
        doc_name = safe_names[doc_index]
        unique_keywords = keyword_set - set.union(
            *[
                other_set
                for other_index, other_set in enumerate(keyword_sets)
                if other_index != doc_index
            ]
        )

        keyword_rows.append(
            {
                "Document": doc_name,
                "Shared keywords": ", ".join(sorted(common_keywords)) or "None detected",
                "Distinctive keywords": ", ".join(sorted(unique_keywords)) or "None detected",
            }
        )

    keyword_table = pd.DataFrame(keyword_rows)

    return {
        "table": comparison_table,
        "details": details,
        "keyword_table": keyword_table,
        "note": (
            "This comparison is extractive and evidence-based. "
            "Each cell is built from sentences detected in the uploaded documents."
        ),
    }

def find_domain_hints(text: str, max_snippets_per_category: int = 4) -> Dict[str, List[str]]:
    """Find short snippets related to common materials-science paper sections."""
    sentences = split_sentences(text)
    hints: Dict[str, List[str]] = {}

    for category, keywords in DOMAIN_KEYWORDS.items():
        category_hits = []
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(keyword in lower_sentence for keyword in keywords):
                category_hits.append(sentence)
            if len(category_hits) >= max_snippets_per_category:
                break
        hints[category] = category_hits

    return hints
