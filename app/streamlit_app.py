"""
Retail AI Co-Pilot — Streamlit application.

UI-only redesign around the existing backend services.
The model, training, authentication, forecasting, pricing, recommendation,
insight and assistant logic remain in the api/ package.
"""
import os
import sys
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.generic_service import GenericStoreService, GenericModelNotFoundError
from api.generic_training import train_store_model, InsufficientDataError
from api.assistant import answer_question, KNOWN_QUESTIONS
from api import accounts
from api.i18n import t, LANGUAGES, is_rtl, RTL_CSS, translate_backend_message

st.set_page_config(
    page_title="Retail AI — Pricing Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design system — deliberately static CSS. No f-string CSS, no fragile hacks.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #687386;
        --line: #e7eaf0;
        --surface: #ffffff;
        --canvas: #f6f7f9;
        --accent: #5b4bdb;
        --accent-soft: #efedff;
        --success: #087f5b;
        --success-soft: #e8f7f1;
        --warning: #b54708;
        --warning-soft: #fff4e5;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }
    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
        letter-spacing: -0.02em;
    }
    #MainMenu, footer { visibility: hidden; }
    /* Keep the Streamlit header available because it contains the sidebar
       expand/collapse control. */
    header { visibility: visible !important; background: transparent !important; }
    header [data-testid="stHeaderActionElements"] { visibility: visible !important; }
    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 0;
    }
    /* Keep the sidebar easy to recover if Streamlit collapses it.
       Do not hide Streamlit's sidebar controls: hiding the collapsed control
       can leave the user with no way to reopen the sidebar. */
    [data-testid="stSidebarCollapseButton"] {
        z-index: 1000;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1000;
    }
    /* Keep the workspace sidebar visible after onboarding/training. */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        visibility: visible !important;
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
        transform: none !important;
        margin-left: 0 !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 21rem !important;
        min-width: 21rem !important;
    }

    section[data-testid="stSidebar"] * { color: #e9edf5; }
    section[data-testid="stSidebar"] .stRadio label {
        border-radius: 10px;
        padding: 0.45rem 0.55rem;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.07);
    }
    section[data-testid="stSidebar"] .stButton > button {
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        color: #fff;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.10);
    }

    .brand {
        padding: 0.25rem 0 1.1rem;
    }
    .brand-mark {
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        background: #6d5ce7;
        color: white;
        font-weight: 800;
        margin-right: 11px;
        vertical-align: middle;
        box-shadow: 0 6px 18px rgba(109,92,231,0.22);
    }
    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 1.04rem;
        font-weight: 800;
        vertical-align: middle;
        color: white;
    }
    .brand-sub {
        color: #aab3c2;
        font-size: 0.74rem;
        margin: 0.55rem 0 0 51px;
        letter-spacing: 0.01em;
    }
    .sidebar-user {
        display: inline-block;
        margin: 0.1rem 0 0.8rem;
        padding: 0.38rem 0.65rem;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 9px;
        background: rgba(255,255,255,0.035);
        color: #b7c0ce;
        font-size: 0.74rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        margin-bottom: 0.35rem;
    }
    .page-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.60rem;
        line-height: 1.1;
        font-weight: 800;
        margin: 0;
    }
    .page-subtitle {
        color: var(--muted);
        margin-top: 0.45rem;
        margin-bottom: 1.35rem;
        font-size: 0.84rem;
    }

    .recommended-metric {
        min-height: 74px;
        height: 74px;
        box-sizing: border-box;
        padding: 0.88rem 1.05rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: 0 2px 8px rgba(16,24,40,0.035);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .recommended-metric-label {
        font-size: 0.74rem;
        line-height: 1.2;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }
    .recommended-metric-value {
        font-family: 'Manrope', sans-serif;
        font-size: 0.92rem;
        line-height: 1.15;
        font-weight: 700;
        color: var(--ink);
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }

    .assistant-intro {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
        padding: 1.05rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(16,24,40,0.035);
    }
    .assistant-intro-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.02rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .assistant-intro-copy {
        color: var(--muted);
        font-size: 0.80rem;
        line-height: 1.45;
    }
    .assistant-source {
        color: var(--success);
        font-size: 0.70rem;
        font-weight: 700;
        margin-top: 0.55rem;
    }

    .status-pill {
        display: inline-block;
        background: var(--success-soft);
        color: var(--success);
        border: 1px solid #c8eddf;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.70rem;
        font-weight: 700;
    }
    .soft-pill {
        display: inline-block;
        background: var(--accent-soft);
        color: #4939bd;
        border-radius: 999px;
        padding: 0.32rem 0.68rem;
        font-size: 0.68rem;
        font-weight: 700;
    }

    .section-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin: 1.4rem 0 0.65rem;
    }
    .section-head h3 {
        margin: 0;
        font-size: 0.94rem;
    }
    .section-head span {
        color: var(--muted);
        font-size: 0.74rem;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 2px 8px rgba(16,24,40,0.035);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 0.92rem !important;
        line-height: 1.18 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: break-word;
        word-break: normal;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem;
    }

    /* Compact typography: keep the existing layout and behavior unchanged. */
    .stMarkdown, .stText, .stCaption, .stAlert, .stExpander,
    .stSelectbox, .stRadio, .stSlider, .stDateInput, .stNumberInput,
    .stTextInput, .stFileUploader {
        font-size: 0.88rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: var(--line);
        border-radius: 14px;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.55rem;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
    }
    .stButton > button[kind="primary"]:hover {
        background: #4939bd;
        border-color: #4939bd;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 0.85rem;
        font-weight: 600;
    }
    .stDataFrame {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

_lang_left, _lang_right = st.columns([6, 1.3])
with _lang_right:
    _lang_choice = st.selectbox(
        t("language_label"),
        options=list(LANGUAGES.keys()),
        format_func=lambda code: LANGUAGES[code],
        index=list(LANGUAGES.keys()).index(st.session_state["lang"]),
        key="lang_selector",
        label_visibility="collapsed",
    )
if _lang_choice != st.session_state["lang"]:
    st.session_state["lang"] = _lang_choice
    st.rerun()

if is_rtl():
    st.markdown(RTL_CSS, unsafe_allow_html=True)


def page_header(title, subtitle, eyebrow="RETAIL INTELLIGENCE", status=None):
    left, right = st.columns([5, 1.2])
    with left:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    if status:
        with right:
            st.markdown(f'<div style="text-align:right;margin-top:0.35rem"><span class="status-pill">● {status}</span></div>', unsafe_allow_html=True)


def logout_button(label="Log out", key="logout"):
    if st.button(label, key=key, use_container_width=True):
        st.session_state["logged_in_user"] = None
        st.session_state.pop("result", None)
        st.session_state.pop("scenarios", None)
        st.rerun()


def render_auth_form():
    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown(f'<div class="eyebrow">{t("app_eyebrow")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-title">{t("app_title")}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="page-subtitle">{t("app_subtitle")}</div>',
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs([t("tab_sign_in"), t("tab_create_account")])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input(t("field_username"), key="auth_username")
                password = st.text_input(t("field_password"), type="password", key="auth_password")
                submitted = st.form_submit_button(t("btn_sign_in"), type="primary", use_container_width=True)
            if submitted:
                if accounts.verify(username, password):
                    st.session_state["logged_in_user"] = username
                    st.rerun()
                else:
                    st.error(t("err_incorrect_login"))

        with register_tab:
            with st.form("register_form"):
                store_name = st.text_input(t("field_business_name"), key="register_store")
                username2 = st.text_input(t("field_username"), key="register_username")
                password2 = st.text_input(t("field_password"), type="password", key="register_password")
                submitted2 = st.form_submit_button(t("btn_create_account"), type="primary", use_container_width=True)
            if submitted2:
                ok, msg = accounts.register(username2, password2, store_name)
                msg = translate_backend_message(msg)
                if ok:
                    st.success(msg + t("msg_account_created_suffix"))
                else:
                    st.error(msg)


def scroll_page_to_top():
    """Reset Streamlit's actual app scroll container after training navigation."""
    components.html(
        """<script>
        (() => {
            const scrollTop = () => {
                try {
                    const parentDoc = window.parent.document;
                    const targets = [
                        parentDoc.querySelector('[data-testid="stAppViewContainer"]'),
                        parentDoc.querySelector('[data-testid="stMain"]'),
                        parentDoc.querySelector('section.main'),
                        parentDoc.documentElement,
                        parentDoc.body,
                    ].filter(Boolean);

                    targets.forEach((el) => {
                        try {
                            el.scrollTo({top: 0, left: 0, behavior: 'auto'});
                            el.scrollTop = 0;
                        } catch (e) {}
                    });

                    try {
                        window.parent.scrollTo(0, 0);
                    } catch (e) {}
                } catch (e) {}
            };

            // Streamlit finishes replacing the page after the rerun, so run
            // after the new DOM is mounted as well as immediately.
            scrollTop();
            [100, 300, 700, 1200, 2000].forEach((delay) => setTimeout(scrollTop, delay));
        })();
        </script>""",
        height=1,
        scrolling=False,
    )


def render_upload_and_train(acc_dir, retrain=False):
    suffix = "retrain" if retrain else "onboarding"
    if not retrain:
        page_header(
            t("hdr_upload_title"),
            t("hdr_upload_subtitle"),
            eyebrow=t("hdr_upload_eyebrow"),
        )

    with st.container(border=True):
        if not retrain:
            st.markdown(f"### {t('upload_section_title')}")
            st.caption(t("upload_section_caption"))
        uploaded = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key=f"upload_{suffix}",
            label_visibility="collapsed",
        )

    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        return

    if df.empty:
        st.error("That file has no rows.")
        return

    none = "(none)"
    with st.container(border=True):
        st.markdown("### Map your columns")
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns detected")
        st.dataframe(df.head(6), use_container_width=True, hide_index=True)
        columns = list(df.columns)
        date_col = st.selectbox("Date column", columns, key=f"date_{suffix}")

        has_units_col = any(c.strip().lower() == "units_sold" for c in columns)
        has_sales_only = any(c.strip().lower() == "sales" for c in columns) and not has_units_col
        if has_sales_only:
            st.info(
                "Your file has a **Sales** column but no **Units_Sold** column. \"Sales\" is "
                "ambiguous -- it could be revenue (price × units) or a raw unit count -- so it "
                "will only be used for sales forecasting below, never assumed to mean units sold "
                "for pricing. If you have real per-product units, price and cost, map them below "
                "to unlock Dynamic Pricing."
            )

        st.markdown("#### Sales forecasting (used for the Forecast / EDA pages)")
        sales_col = st.selectbox(
            "Sales / revenue column", [none] + columns, key=f"sales_{suffix}",
            help="Optional if you map Units_Sold below -- Units_Sold is preferred as the forecasting target.",
        )

        st.markdown("#### Dynamic Pricing columns (Product, Price, Units_Sold, Cost)")
        st.caption(
            "All four are required to enable the Price Simulator, Compare Actions and "
            "Recommendation pages with real, product-aware numbers. Missing any of them "
            "disables Dynamic Pricing with a clear message instead of guessing."
        )
        p1, p2 = st.columns(2)
        product_col = p1.selectbox("Product column", [none] + columns, key=f"product_{suffix}")
        price_col = p2.selectbox("Price column", [none] + columns, key=f"price_{suffix}")
        p3, p4 = st.columns(2)
        units_col = p3.selectbox("Units_Sold column", [none] + columns, key=f"units_{suffix}")
        cost_col = p4.selectbox("Cost column", [none] + columns, key=f"cost_{suffix}")

        with st.expander("Other optional columns"):
            c1, c2, c3 = st.columns(3)
            promo_col = c1.selectbox("Promotion flag", [none] + columns, key=f"promo_{suffix}")
            holiday_col = c2.selectbox("Holiday flag", [none] + columns, key=f"holiday_{suffix}")
            store_col = c3.selectbox("Store / branch", [none] + columns, key=f"store_{suffix}")

    with st.container(border=True):
        st.markdown("### Train model")
        st.caption("Trains 3 candidate models (Random Forest, Extra Trees, Gradient Boosting) and keeps the one with the best validation accuracy.")
        if st.button("Train model", type="primary", key=f"train_{suffix}", use_container_width=True):
            if sales_col == none and units_col == none:
                st.error("Select a Sales/revenue column or a Units_Sold column to train on.")
                return
            with st.spinner("Training on your store data..."):
                try:
                    cfg = train_store_model(
                        df,
                        date_col=date_col,
                        sales_col=None if sales_col == none else sales_col,
                        promo_col=None if promo_col == none else promo_col,
                        holiday_col=None if holiday_col == none else holiday_col,
                        store_col=None if store_col == none else store_col,
                        product_col=None if product_col == none else product_col,
                        price_col=None if price_col == none else price_col,
                        cost_col=None if cost_col == none else cost_col,
                        units_col=None if units_col == none else units_col,
                        output_dir=acc_dir,
                    )
                except (InsufficientDataError, ValueError) as exc:
                    st.error(str(exc))
                    return
            st.success(
                f"Model trained on {cfg['rows_trained']:,} rows · target: {cfg['target']} · "
                f"best model: {cfg['model_name']} · MAE {cfg['metrics']['MAE']:.2f} · RMSE {cfg['metrics']['RMSE']:.2f}"
            )
            if cfg["pricing_enabled"]:
                st.success("Dynamic Pricing is enabled -- Price Simulator, Compare Actions and Recommendations will use your real product data.")
            else:
                st.warning(cfg["pricing_disabled_reason"])
            # The next rerun opens the newly trained workspace. Remember to
            # reset the browser viewport so the new page starts at the top.
            st.session_state["scroll_to_top_after_training"] = True
            st.cache_resource.clear()
            st.rerun()


@st.cache_resource(show_spinner="Loading your store model...")
def load_generic_service(acc_dir):
    return GenericStoreService(acc_dir)


def run_analysis(service, store, forecast_date, objective, base_price=None, elasticity=None,
                  unit_cost=None, product=None, promo=0, holiday=0):
    try:
        dt = datetime(forecast_date.year, forecast_date.month, forecast_date.day)
        if service.pricing_enabled:
            result = service.recommend_for_product(product, store, dt, objective, promo=promo, holiday=holiday)
            scenarios, warning = service.simulate_pricing_for_product(product, store, dt, promo=promo, holiday=holiday)
            st.session_state["pricing_warning"] = warning
            st.session_state["last_inputs"] = dict(
                store=store, date=forecast_date, objective=objective,
                product=product, promo=promo, holiday=holiday,
            )
        else:
            result = service.recommend(store, dt, objective, base_price, elasticity, unit_cost)
            # Use the derived unit estimate (revenue forecast / base_price), not
            # the raw revenue forecast -- simulate_pricing multiplies by price
            # itself, so passing revenue in again double-counts price.
            scenarios = service.simulate_pricing(base_price, elasticity, result["baseline_units"], unit_cost)
            st.session_state["pricing_warning"] = None
            st.session_state["last_inputs"] = dict(
                store=store, date=forecast_date, objective=objective,
                base_price=base_price, elasticity=elasticity, unit_cost=unit_cost,
            )
        st.session_state["result"] = result
        st.session_state["scenarios"] = scenarios
        return True
    except ValueError as exc:
        st.error(str(exc))
        return False


def render_sidebar(service, username, store_name):
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">R</span><span class="brand-name">Retail AI</span>'
            '<div class="brand-sub">Pricing intelligence platform</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="sidebar-user">{store_name} · @{username}</div>', unsafe_allow_html=True)
        st.markdown("---")

        nav_values = [
            "Dashboard",
            "Forecast",
            "Price Simulator",
            "Compare Actions",
            "Recommendations",
            "Business Insights",
            "AI Assistant",
        ]
        nav_labels = {
            "Dashboard": t("nav_dashboard"),
            "Forecast": t("nav_forecast"),
            "Price Simulator": t("nav_pricing"),
            "Compare Actions": t("nav_compare"),
            "Recommendations": t("nav_recommend"),
            "Business Insights": t("nav_insights"),
            "AI Assistant": t("nav_assistant"),
        }
        nav = st.radio(
            t("nav_menu_label"),
            nav_values,
            format_func=lambda v: nav_labels[v],
            key="navigation",
            label_visibility="visible",
        )

        st.markdown("---")
        with st.expander(t("model_management_expander")):
            if st.button(t("upload_retrain_btn"), key="sidebar_model", use_container_width=True):
                st.session_state["open_model_management"] = True
                st.rerun()
        if st.button(t("logout"), key="sidebar_logout", use_container_width=True):
            st.session_state["logged_in_user"] = None
            st.rerun()

        return nav


def render_dashboard(service, store_name):
    metrics = service.model_info["metrics"]
    insights = service.insights or {}
    overview = insights.get("overview", {})

    page_header(
        t("dashboard_greeting", store_name=store_name),
        t("dashboard_subtitle"),
        status=t("model_ready_status"),
    )

    target_name = service.feature_config.get("target", "Sales")
    avg_target = overview.get("avg_target_all", overview.get("avg_sales_all"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Average {target_name}", f"{avg_target:,.2f}" if avg_target is not None else "—")
    c2.metric("Rows trained", f"{service.feature_config.get('rows_trained', 0):,}")
    c3.metric("Model R²", f"{metrics.get('R2', 0):.3f}" if metrics.get("R2") is not None else "—")
    c4.metric("Model MAE", f"{metrics.get('MAE', 0):,.2f}" if metrics.get("MAE") is not None else "—")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Best model", service.feature_config.get("model_name", "—"))
    c6.metric("Number of products", overview.get("num_products") if overview.get("num_products") is not None else "—")
    c7.metric("Number of stores", overview.get("num_stores", "—"))
    if service.pricing_enabled:
        c8.metric("Total revenue", f"{overview.get('total_revenue', 0):,.2f} EGP")
    else:
        c8.metric("Dynamic Pricing", "Disabled")
    if not service.pricing_enabled and service.pricing_disabled_reason:
        st.caption(f"ℹ️ {service.pricing_disabled_reason}")

    st.markdown('<div class="section-head"><h3>Business overview</h3><span>Your real store data</span></div>', unsafe_allow_html=True)
    left, right = st.columns([1.7, 1])

    with left:
        with st.container(border=True):
            tcol = service.feature_config["target"]
            st.markdown(f"### {tcol} history")
            hist = service.history.copy() if service.history is not None else None
            if hist is not None and len(hist) > 1:
                dcol = service.feature_config["date_col"]
                hist[dcol] = pd.to_datetime(hist[dcol])
                # Dashboard shows the aggregate history across all stores/products
                # on purpose; per-product/per-store detail lives on the Forecast page.
                hist_agg = hist.groupby(dcol, as_index=False)[tcol].sum() if service.pricing_enabled else hist
                hist_agg = hist_agg.sort_values(dcol).tail(90)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_agg[dcol], y=hist_agg[tcol], mode="lines",
                    name=tcol, line=dict(color="#5b4bdb", width=2.5),
                    fill="tozeroy", fillcolor="rgba(91,75,219,0.07)",
                ))
                fig.update_layout(
                    height=330, margin=dict(l=8, r=8, t=8, b=8),
                    plot_bgcolor="white", paper_bgcolor="white",
                    hovermode="x unified", showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("History will appear here after your data is available.")

    with right:
        with st.container(border=True):
            st.markdown("### Model health")
            st.markdown('<span class="status-pill">● Trained & loaded</span>', unsafe_allow_html=True)
            st.write("")
            st.caption("Model")
            st.write(service.feature_config.get("model_name", "—"))
            st.caption("Target")
            st.write(str(service.feature_config.get("target", "—")))
            st.caption("Features")
            st.write(str(len(service.feature_config.get("features", []))))
            st.caption("Training rows")
            st.write(f"{service.feature_config.get('rows_trained', 0):,}")

    # Current analysis summary. Navigation is intentionally kept in the sidebar
    # so there is only one place to move between workspace sections.
    st.markdown('<div class="section-head"><h3>Current analysis</h3><span>Based on your latest scenario</span></div>', unsafe_allow_html=True)
    result = st.session_state.get("result")
    inputs = st.session_state.get("last_inputs", {})
    if result and inputs:
        c1, c2, c3, c4 = st.columns(4)
        if service.pricing_enabled:
            baseline = result.get("available_actions", {}).get("No Action", {})
            c1.metric(f"Forecast ({service.feature_config.get('target')})", f"{baseline.get('demand', 0):,.2f}")
            c2.metric("Product", str(result.get("product", "—")))
        else:
            c1.metric("Forecast", f"${result['forecast_sales']:,.2f}")
            c2.metric("Trend", result["forecast_status"])
        c3.metric("Recommended", result["recommended_action"])
        c4.metric("Objective", {
            "balanced": "Balanced",
            "maximize_revenue": "Revenue",
            "maximize_profit": "Profit",
            "maximize_demand": "Demand",
        }.get(inputs.get("objective"), str(inputs.get("objective", "—"))))

        with st.container(border=True):
            st.markdown(f"**Recommended move: {result['recommended_action']}**")
            st.write(result["reason"])
            impact = result.get("expected_impact", {})
            visible_impact = [(k, v) for k, v in impact.items() if k != "ranking"]
            if visible_impact:
                cols = st.columns(min(4, len(visible_impact)))
                for col, (key, value) in zip(cols, visible_impact):
                    display = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
                    col.metric(key.replace("_", " ").title(), display)
    else:
        st.info("Your model is ready. Choose a scenario in the sidebar and update the analysis when you want fresh results.")

    # Quick actions are shortcuts, not a second navigation system. They request
    # the same sidebar navigation and are applied safely on the next rerun.
    st.markdown('<div class="section-head"><h3>What do you want to do?</h3><span>Your analysis is ready to explore</span></div>', unsafe_allow_html=True)
    qa1, qa2, qa3, qa4 = st.columns(4)
    quick_actions = [
        (qa1, "Forecast Sales", "Forecast"),
        (qa2, "Simulate Pricing", "Price Simulator"),
        (qa3, "Compare Actions", "Compare Actions"),
        (qa4, "Get Recommendation", "Recommendations"),
    ]
    for col, label, destination in quick_actions:
        if col.button(label, use_container_width=True, key=f"quick_{destination.lower().replace(' ', '_')}"):
            st.session_state["navigation_override"] = destination
            st.rerun()


def render_forecast(service, inputs):
    target_name = service.feature_config.get("target", "Sales")
    page_header(t("hdr_forecast_title"), t("hdr_forecast_subtitle", target=target_name), eyebrow=t("hdr_forecast_eyebrow"))
    if not inputs:
        st.info(t("set_scenario_info"))
        return
    result = st.session_state.get("result")
    if not result:
        st.info(t("run_to_forecast_info"))
        return

    if service.pricing_enabled:
        baseline = result.get("available_actions", {}).get("No Action", {})
        dt = datetime(inputs["date"].year, inputs["date"].month, inputs["date"].day)
        forecast_value = baseline.get("demand", 0)
        status = service.forecast_status(inputs.get("store"), dt, product=inputs.get("product"))
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Forecasted {target_name}", f"{forecast_value:,.2f}")
        c2.metric("Trend", status)
        c3.metric("Forecast date", str(inputs["date"]))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecasted sales", f"${result['forecast_sales']:,.2f}")
        c2.metric("Trend", result["forecast_status"])
        c3.metric("Forecast date", str(inputs["date"]))
        forecast_value = result["forecast_sales"]

    hist = service.history.copy() if service.history is not None else None
    if hist is not None and len(hist) > 5:
        dcol = service.feature_config["date_col"]
        tcol = service.feature_config["target"]
        hist[dcol] = pd.to_datetime(hist[dcol])
        if service.pricing_enabled and inputs.get("product") and "ProductName" in hist.columns:
            hist = hist[hist["ProductName"] == inputs["product"]]
        hist = hist.sort_values(dcol).tail(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist[dcol], y=hist[tcol], mode="lines", name="Actual", line=dict(color="#344054", width=2)))
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(inputs["date"])], y=[forecast_value],
            mode="markers", name="Forecast", marker=dict(color="#5b4bdb", size=13, symbol="diamond"),
        ))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown(f"### {t('why_forecast_header')}")
        for reason in result["diagnosis"]:
            st.markdown(f"• {reason}")
        st.caption(t("forecast_source_caption"))


def render_pricing(service, scenarios):
    page_header(t("hdr_pricing_title"), t("hdr_pricing_subtitle"), eyebrow=t("hdr_pricing_eyebrow"))
    if not scenarios:
        st.info(t("run_first_info"))
        return
    warning = st.session_state.get("pricing_warning")
    if warning:
        st.warning(warning)
    if service.pricing_enabled:
        st.markdown('<span class="soft-pill">SCENARIOS COMPUTED FROM YOUR PRODUCT\'S ACTUAL PRICE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="soft-pill">SIMULATED FROM AN ASSUMED BASE PRICE</span>', unsafe_allow_html=True)
    df = pd.DataFrame(scenarios)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["scenario"], y=df["expected_revenue"],
        marker_color=["#00A37A" if s == "Baseline" else "#5b4bdb" for s in df["scenario"]],
        name="Expected revenue",
    ))
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Expected revenue ($)")
    st.plotly_chart(fig, use_container_width=True)

    display_cols = ["scenario", "price", "simulated_demand", "expected_revenue", "revenue_change_pct"]
    if "expected_profit" in df.columns:
        display_cols += ["expected_profit", "profit_change_pct"]
    st.dataframe(
        df[display_cols].rename(columns={
            "scenario": "Scenario", "price": "Price ($)", "simulated_demand": "Demand",
            "expected_revenue": "Revenue ($)", "revenue_change_pct": "Revenue Δ%",
            "expected_profit": "Profit ($)", "profit_change_pct": "Profit Δ%",
        }),
        use_container_width=True, hide_index=True,
    )


def render_compare(result):
    page_header(t("hdr_compare_title"), t("hdr_compare_subtitle"), eyebrow=t("hdr_compare_eyebrow"))
    if not result:
        st.info(t("run_first_info"))
        return
    weights = result.get("weights")
    if weights:
        st.caption(
            f"Ranked by a transparent Balanced Score = {weights['revenue_weight']:.0%} × normalized revenue "
            f"+ {weights['profit_weight']:.0%} × normalized profit (both normalized 0–1 across these actions)."
        )
    actions_df = pd.DataFrame(result["available_actions"]).T.sort_values("ranking")
    actions_df.index.name = "Action"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=actions_df.index, x=actions_df["revenue"], orientation="h",
        marker_color=["#00A37A" if r == 1 else "#5b4bdb" for r in actions_df["ranking"]],
    ))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=15, b=10), plot_bgcolor="white", paper_bgcolor="white", xaxis_title="Revenue ($)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(actions_df, use_container_width=True)


def render_recommendation(result):
    page_header(t("hdr_recommend_title"), t("hdr_recommend_subtitle"), eyebrow=t("hdr_recommend_eyebrow"))
    if not result:
        st.info(t("run_first_info"))
        return
    if result.get("current_price") is not None:
        st.caption(
            f"Product: {result.get('product')} · Current price: {result['current_price']:.2f} EGP · "
            f"Current cost: {result['current_cost']:.2f} EGP · Elasticity: {result.get('elasticity_source', '—')}"
        )
    with st.container(border=True):
        st.markdown(f"### {t('recommended_move_header')}")
        st.markdown(f"## {result['recommended_action']}")
        st.write(result["reason"])
    impact = {k: v for k, v in result.get("expected_impact", {}).items() if k != "ranking"}
    if impact:
        st.markdown(f'<div class="section-head"><h3>{t("expected_impact_header")}</h3><span>{t("expected_impact_caption")}</span></div>', unsafe_allow_html=True)
        cols = st.columns(min(4, len(impact)))
        for col, (key, value) in zip(cols, impact.items()):
            display = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
            col.metric(key.replace("_", " ").title(), display)


def render_insights(service):
    page_header(t("hdr_insights_title"), t("hdr_insights_subtitle"), eyebrow=t("hdr_insights_eyebrow"))
    insights = service.insights or {}
    if not insights:
        st.info("No insights available yet.")
        return

    if "overview" in insights:
        ov = insights["overview"]
        avg_target = ov.get("avg_target_all", ov.get("avg_sales_all", 0))
        target_label = ov.get("target_name", "value")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows analyzed", f"{ov['rows_used']:,}")
        c2.metric(f"Average {target_label}", f"{avg_target:,.2f}")
        c3.metric("Date range", f"{ov['date_range'][0]} → {ov['date_range'][1]}")

    if "promotion" in insights:
        p = insights["promotion"]
        st.markdown('<div class="section-head"><h3>Promotion impact</h3><span>Observed in your data</span></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("No promotion", f"${p['avg_no_promo']:,.2f}")
        c2.metric("With promotion", f"${p['avg_with_promo']:,.2f}")
        c3.metric("Sales lift", f"{p['sales_lift_pct']:+.1f}%")

    if "seasonality" in insights:
        s = insights["seasonality"]
        st.markdown('<div class="section-head"><h3>Seasonality</h3><span>Average sales patterns</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow = {names[int(k)-1]: v for k, v in s["avg_by_day_of_week"].items()}
            fig = px.bar(x=list(dow.keys()), y=list(dow.values()))
            fig.update_layout(height=300, margin=dict(l=5,r=5,t=10,b=5), plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            month = {months[int(k)-1]: v for k, v in s["avg_by_month"].items()}
            fig = px.bar(x=list(month.keys()), y=list(month.values()))
            fig.update_layout(height=300, margin=dict(l=5,r=5,t=10,b=5), plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    if "store_performance" in insights:
        sp = insights["store_performance"]
        combined = {**sp["top_5"], **sp["bottom_5"]}
        st.markdown('<div class="section-head"><h3>Store performance</h3><span>Top and bottom locations</span></div>', unsafe_allow_html=True)
        fig = px.bar(x=list(combined.values()), y=list(combined.keys()), orientation="h")
        fig.update_layout(height=350, margin=dict(l=5,r=5,t=10,b=5), plot_bgcolor="white", paper_bgcolor="white", xaxis_title="Average sales ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    if "product_performance" in insights:
        pp = insights["product_performance"]
        st.markdown('<div class="section-head"><h3>Product performance</h3><span>Best and lowest sellers by total units</span></div>', unsafe_allow_html=True)
        combined = {**pp["best_selling"], **pp["lowest_selling"]}
        fig = px.bar(x=list(combined.values()), y=list(combined.keys()), orientation="h")
        fig.update_layout(height=350, margin=dict(l=5,r=5,t=10,b=5), plot_bgcolor="white", paper_bgcolor="white", xaxis_title="Total units sold", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    if "product_pricing" in insights:
        pr = insights["product_pricing"]
        st.markdown('<div class="section-head"><h3>Average price & units by product</h3><span>Observed in your data</span></div>', unsafe_allow_html=True)
        price_df = pd.DataFrame({
            "Product": list(pr["avg_price_by_product"].keys()),
            "Avg price": list(pr["avg_price_by_product"].values()),
            "Avg units sold": [pr["avg_units_by_product"].get(p, None) for p in pr["avg_price_by_product"].keys()],
        })
        st.dataframe(price_df, use_container_width=True, hide_index=True)

    rev = insights.get("product_revenue", {}).get("by_product")
    profit = insights.get("product_profit", {}).get("by_product")
    if rev:
        st.markdown('<div class="section-head"><h3>Revenue & profit by product</h3><span>Sum across your uploaded history</span></div>', unsafe_allow_html=True)
        rp_df = pd.DataFrame({
            "Product": list(rev.keys()),
            "Revenue": list(rev.values()),
            "Profit": [profit.get(p) if profit else None for p in rev.keys()],
        })
        st.dataframe(rp_df, use_container_width=True, hide_index=True)
        if not profit:
            st.caption("Profit is not shown because no Cost column was provided.")

    if "price_demand_relationship" in insights:
        pdr = insights["price_demand_relationship"]
        st.markdown('<div class="section-head"><h3>Price vs. demand relationship</h3><span>Across all products/prices in your data</span></div>', unsafe_allow_html=True)
        if pdr["source"] == "PRIMARY DATA":
            st.metric("Correlation (price vs units sold)", f"{pdr['correlation']:.3f}")
            st.caption(pdr["interpretation"])
        else:
            st.info(pdr["message"])


def render_assistant(service, defaults):
    page_header(
        t("hdr_assistant_title"),
        t("hdr_assistant_subtitle"),
        eyebrow=t("hdr_assistant_eyebrow"),
    )

    st.markdown(
        '<div class="assistant-intro">'
        f'<div class="assistant-intro-title">{t("assistant_intro_title")}</div>'
        f'<div class="assistant-intro-copy">{t("assistant_intro_copy")}</div>'
        '<div class="assistant-source">● Grounded in your current model & scenario</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["chat_history"]:
        st.markdown("**Try asking:**")
        suggestions = [
            "Forecast Sales",
            "Recommend Price",
            "Compare Actions",
            "What should I do next?",
        ]
        cols = st.columns(4)
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, use_container_width=True, key=f"assistant_suggestion_{suggestion}"):
                st.session_state["chat_history"].append(("user", suggestion))
                response = answer_question(
                    suggestion,
                    service,
                    store=defaults.get("store"),
                    date=str(defaults.get("date", date.today())),
                    objective=defaults.get("objective", "balanced"),
                    base_price=defaults.get("base_price", 10.0),
                    elasticity=defaults.get("elasticity", -1.2),
                    unit_cost=defaults.get("unit_cost"),
                    product=defaults.get("product"),
                )
                st.session_state["chat_history"].append(("assistant", response["answer"]))
                st.rerun()

    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input("Ask your retail question…", key="assistant_input")
    if question:
        st.session_state["chat_history"].append(("user", question))
        response = answer_question(
            question,
            service,
            store=defaults.get("store"),
            date=str(defaults.get("date", date.today())),
            objective=defaults.get("objective", "balanced"),
            base_price=defaults.get("base_price", 10.0),
            elasticity=defaults.get("elasticity", -1.2),
            unit_cost=defaults.get("unit_cost"),
            product=defaults.get("product"),
        )
        st.session_state["chat_history"].append(("assistant", response["answer"]))
        st.rerun()


# --------------------------------------------------------------------------
# Authentication + model gate
# --------------------------------------------------------------------------
if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None

if not st.session_state["logged_in_user"]:
    render_auth_form()
    st.stop()

username = st.session_state["logged_in_user"]
store_display_name = accounts.get_store_name(username)
acc_dir = accounts.account_dir(username)

try:
    service = load_generic_service(acc_dir)
except GenericModelNotFoundError:
    render_upload_and_train(acc_dir)
    st.stop()

# After training, Streamlit may preserve the previous browser scroll position.
# Reset it once when the app returns to the main workspace.
if st.session_state.pop("scroll_to_top_after_training", False):
    scroll_page_to_top()

# --------------------------------------------------------------------------
# Sidebar + shared scenario controls
# --------------------------------------------------------------------------
# Apply navigation requests from dashboard quick actions BEFORE the radio
# widget is instantiated. This avoids Streamlit's widget/session-state error.
if "navigation_override" in st.session_state:
    _requested_nav = st.session_state.pop("navigation_override")
    if _requested_nav in [
        "Dashboard", "Forecast", "Price Simulator", "Compare Actions",
        "Recommendations", "Business Insights", "AI Assistant"
    ]:
        st.session_state["navigation"] = _requested_nav

# First-time UX: a trained model should be immediately useful.  Create one
# sensible default analysis automatically instead of forcing a new user to
# understand the sidebar and press Run Analysis before the app has any content.
if "result" not in st.session_state:
    default_store = sorted(service.valid_store_ids)[0] if service.has_store_dimension and service.valid_store_ids else None
    default_date = date.today()
    default_objective = "balanced"
    if service.pricing_enabled:
        default_product = sorted(service.valid_products)[0] if service.valid_products else None
        if run_analysis(service, default_store, default_date, default_objective, product=default_product):
            st.session_state["analysis_initialized"] = True
    else:
        default_price = 10.0
        default_elasticity = -1.2
        default_cost = 6.0
        if run_analysis(service, default_store, default_date, default_objective, default_price, default_elasticity, default_cost):
            st.session_state["analysis_initialized"] = True

# Keep navigation entirely inside the sidebar widget.
# We do not mutate its session state from page content, which avoids
# Streamlit's "widget already instantiated" session-state error.
nav = render_sidebar(service, username, store_display_name)

if st.session_state.pop("open_model_management", False):
    render_upload_and_train(acc_dir, retrain=True)
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown(f"**{t('scenario_header')}**")
    st.caption(t("scenario_caption"))
    if service.has_store_dimension:
        store_options = sorted(service.valid_store_ids)
        store = st.selectbox(t("store_label"), store_options, key="scenario_store")
    else:
        store = None
        st.caption(t("single_location"))

    forecast_date = st.date_input(t("forecast_date_label"), value=date.today(), key="scenario_date")

    product = None
    promo = 0
    holiday = 0
    base_price = None
    elasticity = None
    unit_cost = None

    if service.pricing_enabled:
        objective_labels = {
            "balanced": t("objective_balanced_profit"),
            "maximize_revenue": t("objective_revenue"),
            "maximize_profit": t("objective_profit"),
        }
        objective = st.selectbox(
            t("objective_label"),
            list(objective_labels.keys()),
            format_func=lambda k: objective_labels[k],
            key="scenario_objective",
        )
        product_options = sorted(service.valid_products)
        product = st.selectbox(t("product_label"), product_options, key="scenario_product")
        promo = int(st.checkbox(t("promo_checkbox"), key="scenario_promo"))
        holiday = int(st.checkbox(t("holiday_checkbox"), key="scenario_holiday"))
        try:
            ref = service.get_product_reference(product)
            st.caption(t("current_price_cost_caption", price=ref["current_price"], cost=ref["current_cost"]))
        except ValueError as exc:
            st.error(str(exc))
        current_inputs = dict(store=store, date=forecast_date, objective=objective, product=product, promo=promo, holiday=holiday)
    else:
        objective_labels = {
            "balanced": t("objective_balanced"),
            "maximize_revenue": t("objective_revenue"),
            "maximize_demand": t("objective_demand"),
        }
        objective = st.selectbox(
            t("objective_label"),
            list(objective_labels.keys()),
            format_func=lambda k: objective_labels[k],
            key="scenario_objective",
        )
        with st.expander(t("pricing_assumptions_expander")):
            if service.pricing_disabled_reason:
                st.caption(f"ℹ️ {service.pricing_disabled_reason}")
            base_price = st.number_input(t("base_price_label"), min_value=0.01, value=10.0, step=0.5, key="scenario_price")
            elasticity = st.slider(
                t("elasticity_label"),
                min_value=-3.0, max_value=-0.1, value=-1.2, step=0.1,
                help=t("elasticity_help"),
                key="scenario_elasticity",
            )
            include_cost = st.checkbox(t("include_profit_checkbox"), value=True, key="scenario_profit")
            unit_cost = st.number_input(t("unit_cost_label"), min_value=0.0, value=6.0, step=0.5, key="scenario_cost") if include_cost else None
        current_inputs = dict(store=store, date=forecast_date, objective=objective, base_price=base_price, elasticity=elasticity, unit_cost=unit_cost)

    saved_inputs = st.session_state.get("last_inputs", {})
    scenario_changed = bool(saved_inputs) and any(
        current_inputs.get(k) != saved_inputs.get(k)
        for k in current_inputs
    )
    if scenario_changed:
        st.warning(t("scenario_changed_warning"))
    else:
        st.caption(t("scenario_matches_caption"))

    run_clicked = st.button(t("update_analysis_btn"), type="primary", use_container_width=True, key="run_analysis")
    if st.session_state.get("analysis_updated_at"):
        st.caption(t("last_updated_caption", time=st.session_state["analysis_updated_at"]))

if run_clicked:
    # Recalculate the entire analysis from the current sidebar inputs.
    # Explicitly rerun after a successful calculation so every page/chart
    # immediately reflects the new scenario.
    with st.spinner("Updating your analysis..."):
        updated = run_analysis(
            service, store, forecast_date, objective,
            base_price=base_price, elasticity=elasticity, unit_cost=unit_cost,
            product=product, promo=promo, holiday=holiday,
        )
    if updated:
        st.session_state["analysis_updated"] = True
        st.session_state["analysis_updated_at"] = datetime.now().strftime("%H:%M:%S")
        st.rerun()

result = st.session_state.get("result")
scenarios = st.session_state.get("scenarios")
inputs = st.session_state.get("last_inputs", {})

# --------------------------------------------------------------------------
# Page dispatcher
# --------------------------------------------------------------------------
if nav == "Dashboard":
    render_dashboard(service, store_display_name)
elif nav == "Forecast":
    render_forecast(service, inputs)
elif nav == "Price Simulator":
    render_pricing(service, scenarios)
elif nav == "Compare Actions":
    render_compare(result)
elif nav == "Recommendations":
    render_recommendation(result)
elif nav == "Business Insights":
    render_insights(service)
elif nav == "AI Assistant":
    render_assistant(service, inputs)
