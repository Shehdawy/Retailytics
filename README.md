# Retail AI Co-Pilot

> **We don't just predict what will happen. We help you decide what to do next.**

A self-serve retail forecasting and decision-support tool. Any store creates
an account, uploads their own sales history (any CSV schema), and gets a
real machine learning model trained on their own data in seconds — with
forecasts, price simulations, action comparisons, and a plain-English
recommendation, plus a chat assistant, all in a clean Streamlit dashboard.

There is no built-in sample dataset — every account starts from the store's
own uploaded data.

---

## What It Does

For a store owner, in under a minute:
1. Create an account.
2. Upload a CSV export of your sales history.
3. Get a trained forecasting model, with real accuracy numbers shown immediately.
4. Ask: *what sales should I expect, why, what could I do about it, and what should I actually do?*

## Project Structure

```
project/
├── app/
│   └── streamlit_app.py     # The entire user-facing dashboard
├── api/
│   ├── generic_training.py  # Trains a model on ANY store's uploaded CSV
│   ├── generic_service.py   # Serves forecasts/pricing/decisions for that store
│   ├── assistant.py         # Deterministic AI Assistant (no LLM required)
│   └── accounts.py          # Simple per-store accounts (private data)
├── data/
│   └── customers/           # Created at runtime -- one private folder per account
├── assets/                  # Optional: drop logo.png here for your own branding
├── requirements.txt
├── run_all.py / .bat / .sh  # One-command launch
├── .streamlit/config.toml   # Locked, consistent light theme
├── .env.example
└── .gitignore
```

## Installation & Run

```bash
pip install -r requirements.txt
python run_all.py
```

**Windows:** double-click `run_all.bat`.
**Mac/Linux:** `./run_all.sh`.

Or directly:
```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

## How To Use It

1. **Create an account** — store name, username, password.
2. **Upload your sales CSV.** Map which column is the date and which is
   sales/revenue. Optionally map a promotion flag, a holiday flag, and a
   store/branch column if you have multiple locations.
3. **Click Train My Model.** A Random Forest model trains on your data in
   seconds; you immediately see real accuracy metrics (MAE, RMSE, R²).
4. **Set your scenario** in the sidebar: store/location (if applicable),
   forecast date, business objective, and pricing assumptions.
5. **Click Run Analysis**, then explore the tabs:

| Tab | What you see |
|---|---|
| 📊 Forecast & Diagnosis | Your real sales forecast, trend status, a chart of recent history, and plain-language reasons |
| 💰 Price Simulation | 7 price scenarios (-15% to +15%) with demand/revenue/profit, clearly labeled |
| ⚖️ Compare Actions | 5 business actions ranked side by side |
| 🎯 Recommendation | The best action for your objective, with a reason and expected impact |
| 📈 Business Insights | Real, derived insights from your own data — promotions, seasonality, store performance |
| 🤖 Ask the Assistant | Chat-style Q&A over the same real backend |

6. **Retrain any time** with new data via the "🔄 Upload new data / retrain" section.

## The Honesty Principle

Since no real per-unit price or elasticity exists in typical sales exports,
every number is labeled by where it came from:

| Label | Meaning |
|---|---|
| `PRIMARY DATA` | Computed directly from your uploaded data |
| `ASSUMPTION` | A disclosed, adjustable business assumption (price, elasticity, unit cost) — not measured from your data |
| `SIMULATED` | An output computed from your real forecast + a disclosed assumption |

You can see and adjust every assumption in the sidebar — nothing is hidden.

## Data Privacy

Each account's uploaded data and trained model live in their own folder
(`data/customers/<username>/`) and are only ever loaded for that account.
Passwords are hashed (PBKDF2 + per-account salt), never stored in plain text.
See `api/README.md` for exactly how this works.

## Custom Branding

Drop a `logo.png` file into the `assets/` folder and the app will
automatically display it in the header instead of the default icon — no
code changes needed.

## Turning This Into a Real Product You Can Sell

This section is deliberately honest about what exists today (a working MVP)
versus what a production SaaS product selling to real stores would need.

### What Works Today

- A genuine, working self-serve onboarding flow: any store uploads their own
  data, in any schema, and gets a real trained model with real accuracy
  metrics — not a mockup.
- Each account's data is isolated in its own folder and password-protected.
- The same forecasting/pricing/decision/insights/assistant tools work on
  whatever data a store uploads, whether it's one location or many.

### What This Is NOT Yet (be upfront before selling it)

- **Not enterprise-grade security.** Passwords are hashed, but there's no
  HTTPS enforced by the app itself, no rate limiting on login attempts, no
  email verification or password reset, and accounts are stored in a single
  local JSON file rather than a real database. Fine for a handful of pilot
  customers on a server you control; not fine at scale.
- **Not horizontally scalable.** Everything lives on local disk on whatever
  machine runs `streamlit run`. There's no cloud storage (S3), no database
  (Postgres), and no background job queue — training runs synchronously in
  the browser session, fine for up to a few hundred thousand rows but would
  block the UI for very large uploads.
- **Not billed.** No subscription/payment integration, no plan tiers.
- **Not legally covered.** No Terms of Service, Privacy Policy, or data
  processing agreement — needed before handling real customers' business data.

### A Realistic Path From Here to a Sellable Product

1. **Pilot it as-is, self-hosted, with 1-5 trusted stores.** Run it on a
   VPS you control, behind a domain with free HTTPS (Caddy or nginx +
   Let's Encrypt).
2. **Swap local JSON/disk storage for a real database + object storage**
   once you have more than a handful of customers: PostgreSQL for accounts,
   S3-compatible storage for uploaded CSVs and trained models.
3. **Move training off the request thread** — a background worker (Celery +
   Redis, or similar) so large uploads don't block the UI, and so you can
   retrain on a schedule as new sales data comes in.
4. **Add proper auth** — email verification, password reset, optionally SSO
   — via an auth provider (Auth0, Clerk, Supabase Auth).
5. **Add billing** — Stripe Checkout + subscription tiers.
6. **Get your legal basics in place** — Terms of Service, Privacy Policy,
   and a clear data retention/deletion policy.
7. **Consider a custom frontend** once the product direction is validated —
   Streamlit is excellent for this MVP stage but isn't typically the final
   polished product companies ship at scale.

None of this changes the core value proposition, which already works today:
upload your sales history, get a real forecast, get a real recommendation,
in under a minute.

## Assumptions & Limitations

- Random Forest is used for every account (no model selection UI) — it
  performs reasonably well with no tuning across a wide range of retail
  data, appropriate for a self-serve flow where the user isn't expected to
  be a data scientist.
- Needs at least 30 usable rows of historical sales to train — smaller
  uploads are rejected with a clear message rather than producing an
  unreliable model silently.
- Pricing/elasticity/promo-lift/action-comparison assumptions are the same
  disclosed defaults used throughout the project (see "The Honesty
  Principle" above) — adjust them in the sidebar to match your business.
- The Streamlit UI's visuals were built and syntax-verified in a sandbox
  where `streamlit` itself could not be installed (no internet access) — it
  hasn't been visually verified in a running browser. The underlying logic
  (training, forecasting, recommending, accounts) was fully tested directly
  and works correctly with real data.
