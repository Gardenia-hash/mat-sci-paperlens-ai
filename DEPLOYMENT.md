# Deploy to Streamlit Community Cloud

The repository is prepared for a public Streamlit Community Cloud demo. Deployment requires the repository owner to authorize Streamlit with GitHub in a browser.

## Deployment coordinates

- Repository: `Gardenia-hash/mat-sci-paperlens-ai`
- Branch: `main`
- Entrypoint: `app.py`
- Recommended Python: `3.12`
- Secrets: none required
- Suggested subdomain: `matsci-paperlens-ai`

## One-time deployment

1. Merge the deployment-preparation pull request into `main`.
2. Open [share.streamlit.io](https://share.streamlit.io/) and continue with GitHub.
3. Authorize Streamlit to access the public repository if prompted.
4. Select **Create app** and then **Yup, I have an app**.
5. Enter the repository, branch, and entrypoint shown above.
6. Choose the suggested subdomain if it is available.
7. Open **Advanced settings**, select Python `3.12`, and leave secrets empty.
8. Click **Deploy** and wait for the health check to complete.

## After deployment

Copy the final `https://<subdomain>.streamlit.app` URL and add it to:

1. the repository homepage field;
2. the README using Streamlit's official app badge;
3. the GitHub repository About section or social posts.

Badge Markdown:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<subdomain>.streamlit.app)
```

## Public-demo guardrails

The committed `.streamlit/config.toml` and application checks provide:

- 25 MB maximum per uploaded file;
- up to 6 files per workspace;
- 60 MB combined upload limit;
- bounded Streamlit caches;
- viewer toolbar mode and hidden internal error details;
- a fixed light theme for consistent screenshots;
- a cross-platform Streamlit health-check script used by CI.

To verify deployment readiness locally:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke_streamlit.py
```

## Troubleshooting

- If dependency installation fails, confirm the deployment uses Python 3.12.
- If the app exceeds resources, reboot it from Community Cloud and inspect cache or upload usage.
- If the repository is not listed, reconnect GitHub from Streamlit's **Linked accounts** settings.
- Do not add API keys: the current public demo is designed to work without secrets.

Official references:

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Community Cloud resource limits](https://docs.streamlit.io/knowledge-base/deploy/resource-limits)
