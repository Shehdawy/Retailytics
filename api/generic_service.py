import json
import os
from datetime import timedelta

import joblib
import pandas as pd


class GenericModelNotFoundError(Exception):
    pass


class GenericStoreService:

    def __init__(self, account_dir: str):
        self.account_dir = account_dir
        self.model = None
        self.feature_config = None
        self.insights = None
        self.history = None
        self.product_reference = {}
        self._load()

    def _load(self):
        cfg_path = os.path.join(self.account_dir, "feature_config.json")
        model_path = os.path.join(self.account_dir, "model.pkl")
        if not (os.path.exists(cfg_path) and os.path.exists(model_path)):
            raise GenericModelNotFoundError(
                "No trained model found for this account yet -- upload your sales data first."
            )
        with open(cfg_path) as f:
            self.feature_config = json.load(f)
        self.model = joblib.load(model_path)

        insights_path = os.path.join(self.account_dir, "business_insights.json")
        if os.path.exists(insights_path):
            with open(insights_path) as f:
                self.insights = json.load(f)

        history_path = os.path.join(self.account_dir, "history.csv")
        if os.path.exists(history_path):
            self.history = pd.read_csv(history_path)

        ref_path = os.path.join(self.account_dir, "product_reference.json")
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                self.product_reference = json.load(f)

    @property
    def model_info(self):
        """Shaped for the dashboard's 'model loaded' summary. The model name
        always reflects whichever candidate actually won during training --
        never a hardcoded label -- so the UI can't show e.g. 'Random Forest'
        when Gradient Boosting was the one actually selected and saved."""
        model_name = self.feature_config.get("model_name", "Unknown model")
        return {
            "model_name": f"{model_name} (trained on your data)",
            "features": self.feature_config["features"],
            "target": self.feature_config["target"],
            "metrics": {k: v for k, v in self.feature_config["metrics"].items() if v is not None},
        }

    @property
    def has_store_dimension(self):
        return bool(self.feature_config.get("has_store_dimension"))

    @property
    def valid_store_ids(self):
        if self.has_store_dimension and self.feature_config.get("store_map"):
            return set(self.feature_config["store_map"].keys())
        return set()

    @property
    def product_aware(self):
        """True when the model was trained with Product + Price + Units_Sold,
        i.e. it can predict how demand responds to a different price for a
        specific product -- even if full Dynamic Pricing (which additionally
        needs Cost for profit) is not enabled."""
        return bool(self.feature_config.get("product_aware"))

    @property
    def pricing_enabled(self):
        """True only when Dynamic Pricing can run end-to-end (Product, Price,
        Units_Sold, and Cost are all present), so revenue AND profit can be
        calculated from real data rather than an assumed unit cost."""
        return bool(self.feature_config.get("pricing_enabled"))

    @property
    def pricing_disabled_reason(self):
        return self.feature_config.get("pricing_disabled_reason")

    @property
    def valid_products(self):
        if self.product_aware and self.feature_config.get("product_map"):
            return set(self.feature_config["product_map"].keys())
        return set()

    def get_product_reference(self, product):
        """Returns the real current price/cost/elasticity info for a product,
        computed at training time from that product's own historical rows.
        Raises a clear, catchable error instead of fabricating a price."""
        if not self.product_aware:
            raise ValueError("This account's model was not trained with product-level data.")
        key = str(product)
        if key not in self.product_reference:
            raise ValueError(f"Product '{product}' has no historical price/cost data on file.")
        return self.product_reference[key]

    def _build_row(self, store, date, promo=0, holiday=0, product=None, price=None):
        row = {
            "Year": date.year, "Month": date.month,
            "Week": date.isocalendar()[1], "Day": date.day,
            "DayOfWeek": date.isoweekday(),
            "IsWeekend": int(date.isoweekday() in (6, 7)),
        }
        if self.feature_config.get("has_promo"):
            row["Promo"] = int(promo)
        if self.feature_config.get("has_holiday"):
            row["Holiday"] = int(holiday)
        if self.has_store_dimension:
            store_map = self.feature_config["store_map"]
            if store is None or str(store) not in store_map:
                raise ValueError(f"Unknown store '{store}' for this account.")
            row["StoreCode"] = store_map[str(store)]
        if self.product_aware:
            product_map = self.feature_config["product_map"]
            if product is None or str(product) not in product_map:
                raise ValueError(f"Unknown product '{product}' for this account.")
            row["ProductCode"] = product_map[str(product)]
            if price is None:
                price = self.get_product_reference(product)["current_price"]
            row["Price"] = float(price)
        features = self.feature_config["features"]
        return pd.DataFrame([[row[f] for f in features]], columns=features)

    def forecast(self, store, date, promo=0, holiday=0, product=None, price=None):
        X = self._build_row(store, date, promo, holiday, product=product, price=price)
        pred = float(self.model.predict(X)[0])
        return max(pred, 0.0)

    def forecast_status(self, store, date, promo=0, product=None, price=None):
        current = self.forecast(store, date, promo=promo, product=product, price=price)
        avg = None
        if self.insights and "overview" in self.insights:
            avg = self.insights["overview"].get("avg_target_all") or self.insights["overview"].get("avg_sales_all")
        if not avg:
            return "Unknown"
        diff_pct = (current / avg - 1) * 100
        if diff_pct > 5:
            return "Increasing"
        if diff_pct < -5:
            return "Decreasing"
        return "Stable"

    def forecast_range(self, store, start_date, end_date, promo=0, product=None, price=None):
        results = []
        d = start_date
        while d <= end_date:
            forecast_val = self.forecast(store, d, promo=promo, product=product, price=price)
            results.append({
                "date": d.strftime("%Y-%m-%d"),
                "forecast_value": round(forecast_val, 2),
                "forecast_status": self.forecast_status(store, d, promo=promo, product=product, price=price),
            })
            d += timedelta(days=1)
        return results

    def diagnose(self, store, date):
        reasons = []
        if date.isoweekday() in (6, 7):
            reasons.append("This date falls on a weekend, which affects sales in your historical data.")
        if self.insights and "seasonality" in self.insights:
            if self.insights["seasonality"].get("best_month") == date.month:
                reasons.append("This is historically your strongest month.")
        if self.feature_config.get("has_promo"):
            reasons.append("Promotions have a measurable effect in your data (see Business Insights).")
        if self.has_store_dimension and self.insights and "store_performance" in self.insights:
            top_stores = self.insights["store_performance"].get("top_5", {})
            if store in top_stores:
                reasons.append(f"'{store}' is historically one of your top-performing locations.")
        if not reasons:
            reasons.append("No strong deviation factors identified for this date.")
        return reasons

    # ---------------------------------------------------------------
    # Legacy (non-product-aware) pricing: base_price/elasticity/unit_cost
    # are user-supplied assumptions, kept only for accounts whose uploaded
    # data has no Product/Price/Units_Sold columns to build real pricing
    # from. Unchanged from the original implementation.
    # ---------------------------------------------------------------
    def simulate_pricing(self, base_price, elasticity, baseline_demand, unit_cost=None):
        scenarios_pct = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
        rows = []
        baseline_revenue = baseline_demand * base_price
        baseline_profit = (base_price - unit_cost) * baseline_demand if unit_cost is not None else None
        for pct in scenarios_pct:
            new_price = base_price * (1 + pct)
            simulated_demand = max(baseline_demand * (1 + elasticity * pct), 0)
            expected_revenue = simulated_demand * new_price
            revenue_change_pct = (expected_revenue / baseline_revenue - 1) * 100 if baseline_revenue else 0.0
            row = {
                "scenario": "Baseline" if pct == 0 else f"{pct*100:+.0f}%",
                "price": round(new_price, 2),
                "simulated_demand": round(simulated_demand, 2),
                "expected_revenue": round(expected_revenue, 2),
                "revenue_change_pct": round(revenue_change_pct, 2),
                "pricing_method": "SIMULATED",
                "elasticity_source": "ASSUMPTION",
            }
            if unit_cost is not None:
                expected_profit = (new_price - unit_cost) * simulated_demand
                profit_change_pct = (expected_profit / baseline_profit - 1) * 100 if baseline_profit else 0.0
                row["expected_profit"] = round(expected_profit, 2)
                row["profit_change_pct"] = round(profit_change_pct, 2)
            rows.append(row)
        return rows

    def compare_actions(self, baseline_demand, base_price, elasticity, unit_cost=None):
        PROMO_LIFT = 0.10  # ASSUMPTION

        def _impact(demand, price):
            revenue = demand * price
            profit = (price - unit_cost) * demand if unit_cost is not None else None
            return revenue, profit

        base_revenue, base_profit = _impact(baseline_demand, base_price)
        actions = {}

        actions["No Action"] = {"demand": round(baseline_demand, 2), "price": base_price, "revenue": round(base_revenue, 2)}

        promo_demand = baseline_demand * (1 + PROMO_LIFT)
        promo_revenue, promo_profit = _impact(promo_demand, base_price)
        actions["Promotion"] = {"demand": round(promo_demand, 2), "price": base_price, "revenue": round(promo_revenue, 2)}

        inc_price = base_price * 1.10
        inc_demand = max(baseline_demand * (1 + elasticity * 0.10), 0)
        inc_revenue, inc_profit = _impact(inc_demand, inc_price)
        actions["Price Increase"] = {"demand": round(inc_demand, 2), "price": round(inc_price, 2), "revenue": round(inc_revenue, 2)}

        dec_price = base_price * 0.90
        dec_demand = max(baseline_demand * (1 + elasticity * -0.10), 0)
        dec_revenue, dec_profit = _impact(dec_demand, dec_price)
        actions["Price Decrease"] = {"demand": round(dec_demand, 2), "price": round(dec_price, 2), "revenue": round(dec_revenue, 2)}

        combo_demand = max(baseline_demand * (1 + PROMO_LIFT) * (1 + elasticity * -0.05), 0)
        combo_price = base_price * 0.95
        combo_revenue, combo_profit = _impact(combo_demand, combo_price)
        actions["Promotion + Price Change"] = {"demand": round(combo_demand, 2), "price": round(combo_price, 2), "revenue": round(combo_revenue, 2)}

        profits = {"No Action": base_profit, "Promotion": promo_profit, "Price Increase": inc_profit,
                   "Price Decrease": dec_profit, "Promotion + Price Change": combo_profit}

        for name, data in actions.items():
            data["revenue_change_pct"] = round((data["revenue"] / base_revenue - 1) * 100, 2) if base_revenue else 0.0
            if unit_cost is not None:
                data["profit"] = round(profits[name], 2)
                data["profit_change_pct"] = round((profits[name] / base_profit - 1) * 100, 2) if base_profit else 0.0

        ranked = sorted(actions.items(), key=lambda kv: kv[1]["revenue"], reverse=True)
        for rank, (name, _) in enumerate(ranked, start=1):
            actions[name]["ranking"] = rank

        return actions

    def recommend(self, store, date, objective, base_price, elasticity, unit_cost=None):
        # The trained model predicts REVENUE (the sales/revenue column the
        # store owner mapped at upload time), not a unit count. Pricing math
        # (price x quantity = revenue) needs a quantity, so we back it out
        # from the revenue forecast using the assumed base_price. This is a
        # DERIVED figure, not a second real measurement -- if base_price is
        # far from the store's real average price, baseline_units will be
        # off by the same ratio. (This whole legacy path only runs when the
        # account's data has no real Product/Price/Units_Sold to work with;
        # see recommend_for_product for the product-aware, data-driven path.)
        forecast_sales = self.forecast(store, date)
        baseline_units = (forecast_sales / base_price) if base_price else 0.0
        status = self.forecast_status(store, date)
        reasons = self.diagnose(store, date)
        actions = self.compare_actions(baseline_units, base_price, elasticity, unit_cost)

        if objective == "maximize_demand":
            key = "demand"
            best_action = max(actions.items(), key=lambda kv: kv[1]["demand"])
            score_label = "demand"
        elif objective == "maximize_revenue":
            key = "revenue"
            best_action = max(actions.items(), key=lambda kv: kv[1]["revenue"])
            score_label = "revenue"
        else:
            # Balanced objective: normalize revenue and demand across the
            # available actions, then give both equal weight.
            revenue_values = [v["revenue"] for v in actions.values()]
            demand_values = [v["demand"] for v in actions.values()]
            rev_min, rev_max = min(revenue_values), max(revenue_values)
            dem_min, dem_max = min(demand_values), max(demand_values)
            for data in actions.values():
                rev_score = (data["revenue"] - rev_min) / (rev_max - rev_min) if rev_max != rev_min else 1.0
                dem_score = (data["demand"] - dem_min) / (dem_max - dem_min) if dem_max != dem_min else 1.0
                data["balanced_score"] = round((rev_score + dem_score) / 2, 4)
            best_action = max(actions.items(), key=lambda kv: kv[1]["balanced_score"])
            key = "balanced_score"
            score_label = "balanced score"

        recommended_action, impact = best_action
        reason = (
            f"'{recommended_action}' has the strongest {score_label} ({impact[key]:.2f}) "
            f"among the compared actions for objective '{objective}', given a baseline "
            f"forecast of {forecast_sales:.2f} in sales (~{baseline_units:.2f} units at "
            f"${base_price:.2f}, {status})."
        )

        return {
            "store": store,
            "date": date.strftime("%Y-%m-%d"),
            "forecast_sales": round(forecast_sales, 2),
            "baseline_units": round(baseline_units, 2),
            "units_source": "DERIVED (forecast_sales / base_price)",
            "forecast_status": status,
            "diagnosis": reasons,
            "available_actions": actions,
            "recommended_action": recommended_action,
            "expected_impact": impact,
            "reason": reason,
        }

    # ---------------------------------------------------------------
    # Product-aware pricing: every number below is built from the
    # selected product's OWN historical price/cost/units data, never a
    # hardcoded or user-guessed base_price. Only available when
    # `pricing_enabled` is True.
    # ---------------------------------------------------------------
    def simulate_pricing_for_product(self, product, store, date, promo=0, holiday=0):
        """
        Simulates -15%..+15% price scenarios around the product's actual
        current price. When the product's own historical data has enough
        price variation, demand for each scenario price is taken directly
        from the trained model (which was trained with Price as a real
        feature) -- labeled MODEL PREDICTION. Otherwise, demand is estimated
        from the baseline model prediction using a disclosed elasticity
        assumption -- labeled SIMULATED (ASSUMED ELASTICITY), and a warning
        is returned so the UI can surface it.
        """
        if not self.pricing_enabled:
            raise ValueError(self.pricing_disabled_reason or "Dynamic Pricing is not available for this account.")

        ref = self.get_product_reference(product)
        current_price = ref["current_price"]
        current_cost = ref["current_cost"]
        if current_cost is None:
            raise ValueError(f"'{product}' has no cost data on file, so profit cannot be calculated.")
        elasticity = ref["elasticity"]
        elasticity_source = ref["elasticity_source"]
        price_variation_sufficient = ref["price_variation_sufficient"]

        baseline_units = self.forecast(store, date, promo=promo, holiday=holiday, product=product, price=current_price)
        baseline_revenue = baseline_units * current_price
        baseline_profit = (current_price - current_cost) * baseline_units

        scenarios_pct = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
        rows = []
        for pct in scenarios_pct:
            new_price = round(current_price * (1 + pct), 2)
            if pct == 0.0:
                simulated_units = baseline_units
                method = "MODEL PREDICTION (current price)"
            elif price_variation_sufficient:
                simulated_units = self.forecast(store, date, promo=promo, holiday=holiday, product=product, price=new_price)
                method = "MODEL PREDICTION"
            else:
                simulated_units = max(baseline_units * (1 + elasticity * pct), 0)
                method = "SIMULATED (ASSUMED ELASTICITY)"

            expected_revenue = new_price * simulated_units
            expected_profit = (new_price - current_cost) * simulated_units
            rows.append({
                "scenario": "Baseline" if pct == 0 else f"{pct*100:+.0f}%",
                "price": new_price,
                "simulated_demand": round(simulated_units, 2),
                "expected_revenue": round(expected_revenue, 2),
                "expected_profit": round(expected_profit, 2),
                "revenue_change_pct": round((expected_revenue / baseline_revenue - 1) * 100, 2) if baseline_revenue else 0.0,
                "profit_change_pct": round((expected_profit / baseline_profit - 1) * 100, 2) if baseline_profit else 0.0,
                "pricing_method": method,
                "elasticity_source": elasticity_source,
            })

        warning = None
        if not price_variation_sufficient:
            warning = (
                f"Historical price variation is insufficient to reliably estimate price elasticity "
                f"for '{product}'. The simulation uses an assumed elasticity of {elasticity:.2f}."
            )
        return rows, warning

    def compare_actions_for_product(self, product, store, date, promo=0, holiday=0,
                                     revenue_weight=0.5, profit_weight=0.5):
        """
        Compares 5 realistic business actions using the product's real price
        and cost, and a transparent Balanced Score = revenue_weight *
        normalized(revenue) + profit_weight * normalized(profit), so ranking
        is never based on an arbitrary/undisclosed number.
        """
        if not self.pricing_enabled:
            raise ValueError(self.pricing_disabled_reason or "Dynamic Pricing is not available for this account.")

        ref = self.get_product_reference(product)
        price = ref["current_price"]
        cost = ref["current_cost"]
        if cost is None:
            raise ValueError(f"'{product}' has no cost data on file, so profit cannot be calculated.")
        elasticity = ref["elasticity"]
        price_variation_sufficient = ref["price_variation_sufficient"]

        def units_for(new_price, new_promo):
            if price_variation_sufficient or new_price == price:
                return self.forecast(store, date, promo=new_promo, holiday=holiday, product=product, price=new_price)
            baseline = self.forecast(store, date, promo=new_promo, holiday=holiday, product=product, price=price)
            pct = (new_price / price) - 1
            return max(baseline * (1 + elasticity * pct), 0)

        actions = {}

        def build(name, new_price, new_promo):
            new_price = round(new_price, 2)
            units = units_for(new_price, new_promo)
            revenue = new_price * units
            profit = (new_price - cost) * units
            actions[name] = {
                "price": new_price, "demand": round(units, 2),
                "revenue": round(revenue, 2), "profit": round(profit, 2),
            }

        build("No Action", price, promo)
        build("Promotion", price, 1)
        build("Price Increase", price * 1.10, promo)
        build("Price Decrease", price * 0.90, promo)
        build("Promotion + Price Change", price * 0.95, 1)

        base_revenue = actions["No Action"]["revenue"]
        base_profit = actions["No Action"]["profit"]
        for data in actions.values():
            data["revenue_change_pct"] = round((data["revenue"] / base_revenue - 1) * 100, 2) if base_revenue else 0.0
            data["profit_change_pct"] = round((data["profit"] / base_profit - 1) * 100, 2) if base_profit else 0.0

        revenues = [a["revenue"] for a in actions.values()]
        profits = [a["profit"] for a in actions.values()]
        rev_min, rev_max = min(revenues), max(revenues)
        profit_min, profit_max = min(profits), max(profits)
        for data in actions.values():
            rev_norm = (data["revenue"] - rev_min) / (rev_max - rev_min) if rev_max != rev_min else 1.0
            profit_norm = (data["profit"] - profit_min) / (profit_max - profit_min) if profit_max != profit_min else 1.0
            data["balanced_score"] = round(revenue_weight * rev_norm + profit_weight * profit_norm, 4)

        ranked = sorted(actions.items(), key=lambda kv: kv[1]["balanced_score"], reverse=True)
        for rank, (name, _) in enumerate(ranked, start=1):
            actions[name]["ranking"] = rank

        weights = {"revenue_weight": revenue_weight, "profit_weight": profit_weight}
        return actions, weights

    def recommend_for_product(self, product, store, date, objective, promo=0, holiday=0,
                               revenue_weight=0.5, profit_weight=0.5):
        actions, weights = self.compare_actions_for_product(
            product, store, date, promo, holiday, revenue_weight, profit_weight
        )
        ref = self.get_product_reference(product)
        reasons = self.diagnose(store, date)

        if objective == "maximize_revenue":
            key, label = "revenue", "revenue"
        elif objective == "maximize_profit":
            key, label = "profit", "profit"
        else:
            key, label = "balanced_score", "balanced score"

        best_name, best = max(actions.items(), key=lambda kv: kv[1][key])

        location_clause = f" at {store}" if store else ""
        price_clause = (
            f"Current price is {ref['current_price']:.2f} EGP; this action moves it to {best['price']:.2f} EGP."
            if best["price"] != ref["current_price"] else
            f"This keeps the current price of {ref['current_price']:.2f} EGP."
        )
        reason = (
            f"Among the compared actions for '{product}'{location_clause}, '{best_name}' provides the "
            f"strongest {label} ({best[key]:.2f}) given the demand the model expects at each price point. "
            f"{price_clause}"
        )

        return {
            "product": product,
            "store": store,
            "date": date.strftime("%Y-%m-%d"),
            "current_price": ref["current_price"],
            "current_cost": ref["current_cost"],
            "elasticity_source": ref["elasticity_source"],
            "diagnosis": reasons,
            "available_actions": actions,
            "weights": weights,
            "recommended_action": best_name,
            "expected_impact": best,
            "reason": reason,
        }
