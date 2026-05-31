# Deployment

## GitHub

Initialize and push:

```bash
git init
git add .
git commit -m "Prepare fish pond forecasting project"
gh repo create <repo-name> --public --source=. --remote=origin --push
```

If the repository already exists:

```bash
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```

## Streamlit Community Cloud

This is the recommended deployment path for the current app.

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

The app needs these files in the repository:

- `app.py`
- `requirements.txt`
- `models/meta.json`
- `models/history.parquet`
- `models/global.joblib`
- available `models/pond_*.joblib`
- `model_comparison_v2.csv`

## Vercel

The current project is a Streamlit app. Vercel's Python runtime runs serverless HTTP handlers. It does not run a persistent Streamlit server directly.

To deploy on Vercel, convert the project into:

- A static or framework frontend such as React, Next.js, or plain HTML.
- A Python API endpoint that loads the saved model artifacts and returns predictions.
- A client-side charting layer for forecast and back-test views.

For the current codebase, use Streamlit Community Cloud unless the assignment specifically requires a Vercel URL. If Vercel is mandatory, the dashboard should be converted rather than deployed as-is.
