# 📊 YouTube Analytics Capstone — Streamlit App

Analyze YouTube channels and their videos using the YouTube Data API v3, NLP, and visualizations.

## Quickstart

1) **Create API Key**
   - Create a Google Cloud project → Enable **YouTube Data API v3** → Create an API key.
   - Restrict the key to **YouTube Data API v3** (API restrictions). Keep it secret!

2) **Local Setup**
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
mkdir -p .streamlit
# Add your key to .streamlit/secrets.toml
echo 'YOUTUBE_API_KEY="REPLACE_WITH_YOUR_KEY"' > .streamlit/secrets.toml
streamlit run streamlit_app.py
```

3) **Deploy to Streamlit Cloud**
- Push this folder to GitHub.
- In Streamlit Cloud: **New app** → select repo/branch → main file: `streamlit_app.py`.
- Set **App secrets**: add `YOUTUBE_API_KEY="YOUR_KEY"`.
- Deploy.
