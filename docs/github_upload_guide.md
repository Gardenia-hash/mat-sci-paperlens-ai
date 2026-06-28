# GitHub Upload Guide

Recommended repository name:

```text
mat-sci-paperlens-ai
```

Recommended description:

```text
A lightweight AI/NLP assistant for materials-science papers, with PDF extraction, TF-IDF summarization, keyword extraction, and Streamlit search interface.
```

Recommended topics:

```text
python
streamlit
machine-learning
nlp
materials-science
semiconductor
pdf-processing
scientific-computing
```

## Option A: Upload through GitHub website

1. Go to GitHub.
2. Click **New repository**.
3. Repository name: `mat-sci-paperlens-ai`.
4. Choose **Public**.
5. Do not add README, .gitignore, or license on GitHub because this project already includes them.
6. Create repository.
7. Upload all project files.
8. Commit with:

```text
Initial commit: add MatSci PaperLens AI
```

## Option B: Push with Git command line

Inside this project folder:

```bash
git init
git add .
git commit -m "Initial commit: add MatSci PaperLens AI"
git branch -M main
git remote add origin https://github.com/ZHINIANJIN/mat-sci-paperlens-ai.git
git push -u origin main
```

## After upload

Update your GitHub profile README or CV with:

```text
MatSci PaperLens AI — Built a Python-based AI/NLP literature assistant for materials-science papers, integrating PDF text extraction, TF-IDF summarization, keyword extraction, and query-based passage retrieval in a Streamlit interface.
```
