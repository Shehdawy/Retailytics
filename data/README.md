# data/ — Customer Data

There is no built-in sample dataset in this project. This folder only holds
`customers/`, created automatically at runtime — one private subfolder per
store account (`data/customers/<username>/`), containing:

- The account's `model.pkl`, `feature_config.json`, `business_insights.json`, `history.csv` (see `api/README.md`)
- `data/customers/accounts.json` — hashed account credentials

This folder is listed in `.gitignore` — never commit real customer data.
