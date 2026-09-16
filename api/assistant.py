from datetime import datetime

KNOWN_QUESTIONS = [
    "Forecast Sales",
    "Recommend Price",
    "Compare Actions",
    "Why are sales expected to change?",
    "What should I do next?",
    "Explain Recommendation",
    "Business Insights",
    "Help",
]


def _parse_date(date_str, default=None):
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d")
    return default or datetime.today()


def answer_question(question: str, service, store: int = None, date: str = None,
                     objective: str = "balanced", base_price: float = 10.0,
                     elasticity: float = -1.2, unit_cost: float = None,
                     product: str = None) -> dict:
    """
    `service` is a GenericStoreService instance for the logged-in store --
    always required, since every answer must be traceable to that store's
    own real data and trained model.

    When `service.pricing_enabled` is True, pricing/recommendation questions
    use the product-aware, data-driven path (real price/cost/elasticity for
    `product`) instead of the legacy base_price/elasticity/unit_cost
    assumptions, which only apply to accounts without product-level data.
    """
    q = (question or "").strip().lower()
    dt = _parse_date(date)
    pricing_enabled = bool(getattr(service, "pricing_enabled", False))

    if store is None and getattr(service, "has_store_dimension", True):
        store_needed_keywords = ["forecast", "recommend", "compare", "why", "next", "explain"]
        if any(k in q for k in store_needed_keywords):
            return {
                "question": question,
                "answer": "I need a store number to answer that -- please include 'store' in your request.",
                "source": "assistant_validation",
            }

    if pricing_enabled and product is None:
        product_needed_keywords = ["recommend", "compare", "price", "next", "explain"]
        if any(k in q for k in product_needed_keywords):
            return {
                "question": question,
                "answer": "This account tracks pricing per product -- please select a product first.",
                "source": "assistant_validation",
            }

    if "help" in q:
        return {
            "question": question,
            "answer": "You can ask me: " + "; ".join(KNOWN_QUESTIONS),
            "source": "assistant_static",
        }

    if "business insight" in q or "insight" in q:
        insights = service.insights or {}
        summary_parts = []
        if "promotion" in insights:
            summary_parts.append(f"Promotions are associated with a {insights['promotion']['sales_lift_pct']}% lift.")
        if "seasonality" in insights:
            summary_parts.append(f"The strongest month historically is month {insights['seasonality']['best_month']}.")
        if "product_performance" in insights:
            best = list(insights["product_performance"]["best_selling"].keys())
            if best:
                summary_parts.append(f"Your best-selling product is {best[0]}.")
        answer = " ".join(summary_parts) if summary_parts else "No business insights are available yet."
        return {"question": question, "answer": answer, "source": "business_insights.json"}

    if "forecast" in q and "recommend" not in q:
        forecast_val = service.forecast(store, dt, product=product)
        status = service.forecast_status(store, dt, product=product)
        where = f" for store {store}" if store is not None else ""
        what = f" units of {product}" if pricing_enabled and product else ""
        return {
            "question": question,
            "answer": f"Forecasted{what}{where} on {dt.date()}: {forecast_val:.2f} ({status}).",
            "source": "model_prediction",
        }

    if "why" in q:
        reasons = service.diagnose(store, dt)
        return {
            "question": question,
            "answer": "Possible contributing factors: " + " | ".join(reasons),
            "source": "diagnosis",
        }

    if "recommend price" in q or ("recommend" in q and "price" in q):
        if pricing_enabled:
            scenarios, warning = service.simulate_pricing_for_product(product, store, dt)
            best = max(scenarios, key=lambda s: s["expected_revenue"])
            note = f" Note: {warning}" if warning else ""
            return {
                "question": question,
                "answer": (
                    f"Best-performing simulated price scenario for '{product}': {best['scenario']} "
                    f"(price {best['price']}), expected revenue {best['expected_revenue']}."
                    f" ({best['pricing_method']}){note}"
                ),
                "source": "pricing_simulation",
            }
        # Legacy path: forecast_val is predicted REVENUE, not units -- derive
        # units from the assumed base_price before handing it to the pricing
        # math, otherwise price gets multiplied in twice.
        forecast_val = service.forecast(store, dt)
        baseline_units = (forecast_val / base_price) if base_price else 0.0
        scenarios = service.simulate_pricing(base_price, elasticity, baseline_units, unit_cost)
        best = max(scenarios, key=lambda s: s["expected_revenue"])
        return {
            "question": question,
            "answer": f"Best-performing simulated price scenario: {best['scenario']} (price {best['price']}), "
                      f"expected revenue {best['expected_revenue']}. (pricing_method=SIMULATED, elasticity_source=ASSUMPTION)",
            "source": "pricing_simulation",
        }

    if "compare action" in q or ("compare" in q and "action" in q):
        if pricing_enabled:
            actions, _weights = service.compare_actions_for_product(product, store, dt)
            ranked = sorted(actions.items(), key=lambda kv: kv[1]["ranking"])
            lines = [f"{name}: revenue {data['revenue']} / profit {data['profit']} (rank {data['ranking']})" for name, data in ranked]
            return {"question": question, "answer": "; ".join(lines), "source": "decision_engine"}
        forecast_val = service.forecast(store, dt)
        baseline_units = (forecast_val / base_price) if base_price else 0.0
        actions = service.compare_actions(baseline_units, base_price, elasticity, unit_cost)
        ranked = sorted(actions.items(), key=lambda kv: kv[1]["ranking"])
        lines = [f"{name}: revenue {data['revenue']} (rank {data['ranking']})" for name, data in ranked]
        return {"question": question, "answer": "; ".join(lines), "source": "decision_engine"}

    if "what should i do" in q or "next" in q:
        if pricing_enabled:
            result = service.recommend_for_product(product, store, dt, objective)
        else:
            result = service.recommend(store, dt, objective, base_price, elasticity, unit_cost)
        return {
            "question": question,
            "answer": f"Recommended Action: {result['recommended_action']}. Reason: {result['reason']}",
            "source": "decision_engine",
        }

    if "explain" in q:
        if pricing_enabled:
            result = service.recommend_for_product(product, store, dt, objective)
        else:
            result = service.recommend(store, dt, objective, base_price, elasticity, unit_cost)
        return {
            "question": question,
            "answer": result["reason"],
            "source": "decision_engine",
        }

    return {
        "question": question,
        "answer": "I didn't recognize that question. Try: " + "; ".join(KNOWN_QUESTIONS),
        "source": "assistant_fallback",
    }
