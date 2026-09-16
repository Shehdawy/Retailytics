# api/ — Business Logic Layer

Everything here is plain Python, called directly by `app/streamlit_app.py`.
No web server, no REST endpoints — the dashboard imports these functions
in-process.

```
api/
├── generic_training.py  # Trains a Random Forest on ANY store's uploaded CSV
├── generic_service.py   # GenericStoreService -- forecasting/pricing/decisions
├── assistant.py         # Deterministic AI Assistant, no LLM required
└── accounts.py          # Simple per-store accounts (private data)
```

## `generic_training.py`

`train_store_model(df, date_col, sales_col, promo_col=None, holiday_col=None, store_col=None, output_dir=...)`

Trains a `RandomForestRegressor` on whatever columns the user mapped —
requires at least a date and a sales/revenue column. Engineers calendar
features (year, month, week, day, day-of-week, weekend flag) plus whichever
optional columns were mapped. Uses a time-aware split (last ~15% of rows for
validation, not random) so evaluation reflects real forecasting performance.
Writes to `output_dir`:
- `model.pkl` — the trained model
- `feature_config.json` — feature list, target, real metrics (MAE/RMSE/R²), whether the account has multiple store locations
- `business_insights.json` — real, derived insights (promotion lift, seasonality, store performance if multi-location)
- `history.csv` — cleaned date+sales history, used for the dashboard's trend chart

Raises `InsufficientDataError` if fewer than 30 usable rows remain after cleaning.

## `generic_service.py` — `GenericStoreService`

Loads one account's artifacts and serves:

| Method | Purpose |
|---|---|
| `forecast(store, date, promo=0, holiday=0)` | Real model prediction |
| `forecast_status(store, date, promo=0)` | Increasing / Decreasing / Stable vs. this store's average |
| `forecast_range(store, start_date, end_date, promo=0)` | Forecasts across a date range |
| `diagnose(store, date)` | Plain-language reasons behind the forecast (weekend, best month, promo effect, top-performing location) |
| `simulate_pricing(base_price, elasticity, baseline_demand, unit_cost=None)` | 7-scenario what-if price simulation |
| `compare_actions(baseline_demand, base_price, elasticity, unit_cost=None)` | Compares 5 fixed business actions |
| `recommend(store, date, objective, base_price, elasticity, unit_cost=None)` | Full predict → diagnose → simulate → compare → recommend pipeline |

`store` is `None` for single-location accounts (no store column was mapped).

## `assistant.py`

`answer_question(question, service, store=None, date=None, objective="balanced", base_price=10.0, elasticity=-1.2, unit_cost=None)`
— recognizes a fixed set of questions (see `KNOWN_QUESTIONS`) and answers
using the real `GenericStoreService` methods above. `service` is required —
every answer must be traceable to that account's own real data.

## `accounts.py`

Simple username/password accounts, hashed with PBKDF2-HMAC-SHA256 + a
per-account random salt. Each account's model/data folder is
`data/customers/<username>/`, created and used only for that account.

**MVP-level security note:** this is reasonable for a small pilot, not a
substitute for a real auth provider before selling at scale — see the root
`README.md`'s "Turning This Into a Real Product" section.
