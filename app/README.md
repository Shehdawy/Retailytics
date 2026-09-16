# app/ — Streamlit Dashboard

The entire user-facing interface, in `streamlit_app.py`. Thin presentation
layer only — every number comes from `api/generic_service.py`,
`api/generic_training.py`, and `api/assistant.py`.

## Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

## Flow

1. **Landing / Auth** — hero section, feature highlights, log in or create an account.
2. **Upload & Train** (first time only) — upload a CSV, map columns, train.
3. **Dashboard** — sidebar scenario inputs + 6 tabs:

| Tab | What you see |
|---|---|
| 📊 Forecast & Diagnosis | Forecast, trend status, a real chart of recent sales history, plain-language reasons |
| 💰 Price Simulation | 7 price scenarios, bar chart + table |
| ⚖️ Compare Actions | 5 actions ranked, horizontal bar chart |
| 🎯 Recommendation | Highlighted recommendation card + expected impact metrics |
| 📈 Business Insights | Charts for promotion lift, seasonality, store performance |
| 🤖 Ask the Assistant | Chat interface |

## Design notes

- Custom CSS gives a consistent branded look (gradient hero, card-style
  metrics, Poppins/Inter fonts) regardless of the user's system theme —
  `.streamlit/config.toml` locks the base theme so it never breaks due to a
  visitor's dark-mode setting.
- Charts use Plotly for a polished, interactive look instead of Streamlit's
  built-in basic charts.
- Drop `assets/logo.png` in the project to show your own branding instead of
  the default hero icon — no code changes needed.
- Model loading is cached (`st.cache_resource`) per account, so the app
  stays fast across interactions instead of reloading a model on every click.
