# Project Creation Plan

## Project name

MatSci PaperLens AI

## One-sentence goal

Build a local AI/NLP assistant that helps screen materials-science and semiconductor papers by extracting summaries, keywords, relevant passages, and domain-specific hints.

## Why this project fits the author

The project combines:

- materials science and semiconductor domain knowledge
- Python scientific computing
- machine learning / NLP
- PDF processing
- GitHub portfolio presentation

It is suitable for an incoming Materials Science and Engineering MSc student with interests in semiconductor materials, metrology, failure analysis, characterization, and AI-assisted research workflows.

## MVP checklist

- [x] Create repository structure
- [x] Add Streamlit app
- [x] Add PDF extraction module
- [x] Add TF-IDF summarization
- [x] Add keyword extraction
- [x] Add question-style passage retrieval
- [x] Add materials-science domain hints
- [x] Add sample text
- [x] Add unit tests
- [x] Add GitHub Actions workflow
- [x] Add README and upload guide

## Suggested GitHub issues

### Issue 1: Add multi-paper comparison

Compare two or more uploaded papers and extract differences in material system, method, result, and limitation.

### Issue 2: Add semantic search mode

Add optional Sentence Transformers support so the search can capture semantic similarity beyond keyword overlap.

### Issue 3: Add export function

Allow the user to export summaries, keywords, and search results to Markdown or CSV.

### Issue 4: Add screenshot examples

Run the app locally, take screenshots, and add them to the README.

### Issue 5: Add paper metadata parser

Extract title, authors, abstract, DOI, and publication year when possible.

## Suggested first five commits

```text
Initial commit: add MatSci PaperLens AI project scaffold
Add PDF and text processing utilities
Add TF-IDF summary and keyword extraction
Add Streamlit user interface
Add tests and GitHub Actions workflow
```

## Future advanced direction

This project can evolve into a serious research assistant:

- RAG over a personal paper library
- material-method-result table extraction
- citation graph analysis
- integration with Zotero
- embedding-based clustering of papers
- LLM-generated literature review drafts with citation checking
