# Streamlit Community Cloud deployment

The public demo is live and automatically follows the repository's `main` branch.

## Production URL

[https://mat-sci-paperlens-ai-nesmxsbkrzw5ezrce6z5pu.streamlit.app/](https://mat-sci-paperlens-ai-nesmxsbkrzw5ezrce6z5pu.streamlit.app/)

## Deployment coordinates

- Repository: `Gardenia-hash/mat-sci-paperlens-ai`
- Branch: `main`
- Entrypoint: `app.py`
- Recommended Python: `3.12`
- Secrets: none required
- Current Streamlit subdomain: `mat-sci-paperlens-ai-nesmxsbkrzw5ezrce6z5pu`

## Recreate the deployment

Use these steps only if the existing app is deleted or must be recreated:

1. Open [share.streamlit.io](https://share.streamlit.io/) and continue with GitHub.
2. Authorize Streamlit to access the public repository if prompted.
3. Select **Create app** and then **Yup, I have an app**.
4. Enter the repository, branch, and entrypoint shown above.
5. Choose an available custom subdomain.
6. Open **Advanced settings**, select Python `3.12`, and leave secrets empty.
7. Click **Deploy** and wait for the health check to complete.

## Repository integration

The production URL is published in:

1. the GitHub repository homepage field;
2. the README's official Streamlit app badge;
3. the clickable application screenshot.

Badge Markdown:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mat-sci-paperlens-ai-nesmxsbkrzw5ezrce6z5pu.streamlit.app/)
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
