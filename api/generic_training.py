import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Disclosed default used only when a product's own historical data does not
# contain enough price variation to estimate its own elasticity. This is
# clearly labeled as an ASSUMPTION everywhere it is surfaced -- it is never
# presented as a measured fact.
DEFAULT_ELASTICITY = -1.2
MIN_ROWS_FOR_ELASTICITY = 5
MIN_PRICE_VARIATION_CV = 0.02  # coefficient of variation (std / mean)


class InsufficientDataError(Exception):
    pass


def _estimate_elasticity(sub_df: pd.DataFrame, price_col: str, target_col: str):
    """
    Estimates price elasticity of demand for one product from its own
    historical (price, units_sold) observations using a log-log regression:
    ln(units) = a + b * ln(price). The slope b is the elasticity estimate.

    Returns None when there isn't enough usable data or price variation to
    make that estimate meaningful -- callers must fall back to a disclosed
    assumption in that case, never silently reuse a stale/default number as
    if it were measured.
    """
    prices = pd.to_numeric(sub_df[price_col], errors="coerce")
    units = pd.to_numeric(sub_df[target_col], errors="coerce")
    mask = (prices > 0) & (units > 0)
    prices = prices[mask].values
    units = units[mask].values

    if len(prices) < MIN_ROWS_FOR_ELASTICITY:
        return None

    mean_price = float(np.mean(prices))
    if mean_price <= 0:
        return None
    price_cv = float(np.std(prices) / mean_price)
    if price_cv < MIN_PRICE_VARIATION_CV:
        return None

    try:
        slope, _intercept = np.polyfit(np.log(prices), np.log(units), 1)
    except Exception:
        return None

    if not np.isfinite(slope):
        return None

    return {"value": float(slope), "source": "ESTIMATED FROM DATA (log-log regression on this product's own price/units history)"}


def train_store_model(df: pd.DataFrame, date_col: str, sales_col: str = None,
                       promo_col: str = None, holiday_col: str = None,
                       store_col: str = None, product_col: str = None,
                       price_col: str = None, cost_col: str = None,
                       units_col: str = None, output_dir: str = None) -> dict:
    """
    Trains a demand-forecasting model on a store's own data and writes
    model.pkl, feature_config.json, business_insights.json, history.csv,
    and (when the data supports it) product_reference.json into output_dir.

    Two modes, chosen automatically from which columns are supplied:

    - Product-aware mode (product_col + price_col + units_col all present):
      the model is trained to predict Units_Sold, with Product, Store, Price,
      Promo, Holiday, and date features as inputs. Price is a real feature,
      so the same model can predict how demand responds to a different price
      -- this is what the Price Simulator / Compare Actions / Recommendation
      pages use, instead of a hardcoded or assumed base price. Dynamic
      Pricing (which additionally needs profit) is only enabled when a Cost
      column is also present; otherwise it stays cleanly disabled with an
      informative reason rather than fabricating a cost.

    - Legacy sales-forecasting mode (no Units_Sold / Product / Price): trains
      on the given sales/revenue column, exactly as before. Dynamic Pricing
      is disabled in this mode because there is no real product price/cost
      to build it from.

    Three tree-based regressors are trained on the same split; the one with
    the best validation MAE (RMSE as tie-breaker) is kept as model.pkl, and
    its name is recorded in feature_config["model_name"] so every page shows
    the model that actually won, never a hardcoded one.
    """
    df = df.copy()

    if date_col not in df.columns:
        raise ValueError("The selected date column was not found in the uploaded file.")

    has_units_target = bool(units_col and units_col in df.columns)
    has_sales_target = bool(sales_col and sales_col in df.columns)
    if not has_units_target and not has_sales_target:
        raise ValueError(
            "Select a Units_Sold column (preferred) or a Sales/revenue column to train on."
        )

    has_product = bool(product_col and product_col in df.columns)
    has_price = bool(price_col and price_col in df.columns)
    has_cost = bool(cost_col and cost_col in df.columns)

    # Product-aware mode requires the demand target to genuinely be a unit
    # count. Sales/revenue is ambiguous (it already bakes price in), so it
    # is never substituted for Units_Sold here.
    product_aware = bool(has_product and has_price and has_units_target)

    pricing_enabled = bool(product_aware and has_cost)
    pricing_disabled_reason = None
    if product_aware and not has_cost:
        pricing_disabled_reason = (
            "Dynamic Pricing is disabled: your data has Product, Price and Units_Sold, "
            "but no Cost column, so profit cannot be calculated. Forecasting, price "
            "simulation of revenue, and business insights still work -- add a Cost "
            "column and retrain to unlock profit-aware pricing and recommendations."
        )
    elif not product_aware:
        missing = []
        if not has_product:
            missing.append("Product")
        if not has_price:
            missing.append("Price")
        if not has_units_target:
            missing.append("Units_Sold")
        pricing_disabled_reason = (
            "Dynamic Pricing is disabled: your data is missing " + ", ".join(missing) +
            ". Sales forecasting and business insights still work with what you uploaded."
        )

    target = units_col if has_units_target else sales_col

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    required_clean_cols = [date_col, target]
    if product_aware:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        required_clean_cols.append(price_col)
        if has_cost:
            df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

    df = df.dropna(subset=required_clean_cols)
    df = df[df[target] >= 0]
    if product_aware:
        df = df[df[price_col] > 0]
    df = df.sort_values(date_col)

    if len(df) < 30:
        raise InsufficientDataError(
            f"Only {len(df)} usable rows after cleaning -- need at least 30 rows "
            f"of historical data to train a reliable model."
        )

    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["Day"] = df[date_col].dt.day
    df["DayOfWeek"] = df[date_col].dt.dayofweek + 1
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    features = ["Year", "Month", "Week", "Day", "DayOfWeek", "IsWeekend"]

    has_promo = bool(promo_col and promo_col in df.columns)
    if has_promo:
        df["Promo"] = pd.to_numeric(df[promo_col], errors="coerce").fillna(0).astype(int)
        features.append("Promo")

    has_holiday = bool(holiday_col and holiday_col in df.columns)
    if has_holiday:
        df["Holiday"] = pd.to_numeric(df[holiday_col], errors="coerce").fillna(0).astype(int)
        features.append("Holiday")

    has_store_dimension = bool(store_col and store_col in df.columns)
    store_map = None
    if has_store_dimension:
        df["StoreName"] = df[store_col].astype(str)
        store_categories = sorted(df["StoreName"].unique())
        store_map = {name: i for i, name in enumerate(store_categories)}
        df["StoreCode"] = df["StoreName"].map(store_map)
        features.append("StoreCode")

    product_map = None
    if product_aware:
        df["ProductName"] = df[product_col].astype(str)
        product_categories = sorted(df["ProductName"].unique())
        product_map = {name: i for i, name in enumerate(product_categories)}
        df["ProductCode"] = df["ProductName"].map(product_map)
        features.append("ProductCode")
        df["Price"] = df[price_col]
        features.append("Price")

    n = len(df)
    split_idx = int(n * 0.85)
    split_idx = min(split_idx, n - 1)  # always leave at least 1 row for validation
    split_idx = max(split_idx, 1)      # always leave at least 1 row for training
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target]
    X_valid, y_valid = valid_df[features], valid_df[target]

    # Train three tree-based regressors on the exact same train/validation split.
    # The best model is selected by validation MAE, with RMSE as a tie-breaker.
    candidate_models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=3, random_state=42,
        ),
    }

    model_results = {}
    best_model = None
    best_name = None
    best_key = None

    for name, candidate in candidate_models.items():
        candidate.fit(X_train, y_train)
        candidate_pred = candidate.predict(X_valid)
        candidate_mae = float(mean_absolute_error(y_valid, candidate_pred))
        candidate_rmse = float(np.sqrt(mean_squared_error(y_valid, candidate_pred)))
        candidate_r2 = float(r2_score(y_valid, candidate_pred)) if len(y_valid) > 1 else None

        model_results[name] = {
            "MAE": candidate_mae,
            "RMSE": candidate_rmse,
            "R2": candidate_r2,
        }

        ranking_key = (candidate_mae, candidate_rmse)
        if best_key is None or ranking_key < best_key:
            best_key = ranking_key
            best_model = candidate
            best_name = name

    model = best_model
    mae = model_results[best_name]["MAE"]
    rmse = model_results[best_name]["RMSE"]
    r2 = model_results[best_name]["R2"]

    # ---- Generic, data-derived business insights (no invented numbers) ----
    insights = {}

    if has_promo:
        promo_avg = df.groupby("Promo")[target].mean()
        if 0 in promo_avg.index and 1 in promo_avg.index and promo_avg[0] > 0:
            insights["promotion"] = {
                "source": "PRIMARY DATA",
                "avg_no_promo": round(float(promo_avg[0]), 2),
                "avg_with_promo": round(float(promo_avg[1]), 2),
                "sales_lift_pct": round(float((promo_avg[1] / promo_avg[0] - 1) * 100), 2),
            }

    dow_avg = df.groupby("DayOfWeek")[target].mean().round(2).to_dict()
    month_avg = df.groupby("Month")[target].mean().round(2).to_dict()
    week_avg = df.groupby("Week")[target].mean().round(2).to_dict()
    insights["seasonality"] = {
        "source": "PRIMARY DATA",
        "avg_by_day_of_week": {int(k): v for k, v in dow_avg.items()},
        "avg_by_month": {int(k): v for k, v in month_avg.items()},
        "best_day_of_week": int(max(dow_avg, key=dow_avg.get)),
        "best_month": int(max(month_avg, key=month_avg.get)),
    }
    insights["weekly_trend"] = {
        "source": "PRIMARY DATA",
        "avg_by_iso_week": {int(k): v for k, v in week_avg.items()},
    }

    insights["overview"] = {
        "source": "PRIMARY DATA",
        "rows_used": int(n),
        "date_range": [str(df[date_col].min().date()), str(df[date_col].max().date())],
        "avg_target_all": round(float(df[target].mean()), 2),
        "target_name": target,
        "num_stores": int(df["StoreName"].nunique()) if has_store_dimension else 1,
        "num_products": int(df["ProductName"].nunique()) if product_aware else None,
    }

    if has_store_dimension:
        store_perf = df.groupby("StoreName")[target].mean().sort_values(ascending=False)
        insights["store_performance"] = {
            "source": "PRIMARY DATA",
            "top_5": store_perf.head(5).round(2).to_dict(),
            "bottom_5": store_perf.tail(5).round(2).to_dict(),
            "avg_across_stores": round(float(store_perf.mean()), 2),
        }

    product_reference = {}
    if product_aware:
        product_units = df.groupby("ProductName")[target].sum().sort_values(ascending=False)
        insights["product_performance"] = {
            "source": "PRIMARY DATA",
            "best_selling": product_units.head(5).round(2).to_dict(),
            "lowest_selling": product_units.tail(5).round(2).to_dict(),
        }
        insights["product_pricing"] = {
            "source": "PRIMARY DATA",
            "avg_price_by_product": df.groupby("ProductName")[price_col].mean().round(2).to_dict(),
            "avg_units_by_product": df.groupby("ProductName")[target].mean().round(2).to_dict(),
        }
        revenue_series = df[price_col] * df[target]
        insights["product_revenue"] = {
            "source": "PRIMARY DATA",
            "by_product": revenue_series.groupby(df["ProductName"]).sum().round(2).to_dict(),
        }
        insights["overview"]["total_revenue"] = round(float(revenue_series.sum()), 2)

        if has_cost:
            profit_series = (df[price_col] - df[cost_col]) * df[target]
            insights["product_profit"] = {
                "source": "PRIMARY DATA",
                "by_product": profit_series.groupby(df["ProductName"]).sum().round(2).to_dict(),
            }
            insights["overview"]["total_profit"] = round(float(profit_series.sum()), 2)

        if df[price_col].nunique() > 1:
            corr = float(df[[price_col, target]].corr().iloc[0, 1])
            insights["price_demand_relationship"] = {
                "source": "PRIMARY DATA",
                "correlation": round(corr, 3),
                "interpretation": (
                    "Higher prices are associated with lower units sold in your data."
                    if corr < -0.05 else
                    "Higher prices are associated with higher units sold in your data -- "
                    "unusual, and worth checking for confounding factors (e.g. promotions "
                    "coinciding with price changes)."
                    if corr > 0.05 else
                    "No clear linear relationship between price and units sold was found in your data."
                ),
            }
        else:
            insights["price_demand_relationship"] = {
                "source": "INSUFFICIENT DATA",
                "message": "Only one price point exists in your data, so the price-demand relationship cannot be assessed.",
            }

        # Per-product reference: current price/cost + elasticity, used by the
        # pricing/service layer instead of any hardcoded base_price.
        for prod, sub in df.groupby("ProductName"):
            sub_sorted = sub.sort_values(date_col)
            latest = sub_sorted.iloc[-1]
            estimate = _estimate_elasticity(sub, price_col, target)
            product_reference[prod] = {
                "current_price": round(float(latest[price_col]), 2),
                "current_cost": round(float(latest[cost_col]), 2) if has_cost else None,
                "avg_price": round(float(sub[price_col].mean()), 2),
                "rows": int(len(sub)),
                "elasticity": estimate["value"] if estimate else DEFAULT_ELASTICITY,
                "elasticity_source": estimate["source"] if estimate else (
                    f"ASSUMPTION (insufficient historical price variation for '{prod}'; "
                    f"using a disclosed default of {DEFAULT_ELASTICITY})"
                ),
                "price_variation_sufficient": bool(estimate),
            }

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))

    feature_config = {
        "features": features,
        "target": target,
        "date_col": date_col,
        "has_promo": has_promo,
        "has_holiday": has_holiday,
        "has_store_dimension": has_store_dimension,
        "store_map": store_map,
        "product_aware": product_aware,
        "pricing_enabled": pricing_enabled,
        "pricing_disabled_reason": pricing_disabled_reason,
        "product_map": product_map,
        "default_elasticity": DEFAULT_ELASTICITY,
        "model_name": best_name,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "model_comparison": model_results,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "rows_trained": int(len(train_df)),
        "rows_validated": int(len(valid_df)),
    }
    with open(os.path.join(output_dir, "feature_config.json"), "w") as f:
        json.dump(feature_config, f, indent=2)
    with open(os.path.join(output_dir, "business_insights.json"), "w") as f:
        json.dump(insights, f, indent=2, default=str)
    with open(os.path.join(output_dir, "product_reference.json"), "w") as f:
        json.dump(product_reference, f, indent=2, default=str)

    history_cols = [date_col, target]
    if has_store_dimension:
        history_cols.append("StoreName")
    if product_aware:
        history_cols += ["ProductName", price_col] + ([cost_col] if has_cost else [])
    df[history_cols].to_csv(os.path.join(output_dir, "history.csv"), index=False)

    return feature_config
