"""Artificial Intelligence in Retail and E-commerce: Transforming Customer Experience.
Comprehensive Streamlit Academic Application Dashboard based on Prajapati et al. (2024).

Features:
1. Executive Overview & Architecture Diagram (Section III of paper)
2. Personalized Product Recommendation with Explainable AI (Section IV-B)
3. Conversational AI Customer Support & Order Tracking (Section IV-B)
4. Adaptive Dynamic Pricing with Fairness Safeguards (Section IV-B & VII-B)
5. Machine Learning Evaluation Dashboard (Section VI-A)
6. Research Paper vs. Experimental Results Comparative Analysis (Section VI & Table I)
7. Relational Database & Inventory Explorer (Section III-B)
"""
from pathlib import Path
import json
import sys
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import (
    init_db, load_products_df, load_customers_df, load_orders_df,
    get_order_by_id, search_products, get_pricing_history
)
from src.recommender import get_recommender
from src.chatbot import get_chatbot
from src.dynamic_pricing import get_pricing_engine
from src.evaluation import (
    PAPER_REPORTED_RESULTS, load_experimental_metrics,
    load_confusion_matrices, load_feature_importances,
    build_paper_vs_experimental_comparison,
    plot_metrics_comparison_chart, plot_confusion_matrix_heatmap,
    plot_feature_importance_chart, plot_radar_comparison
)
from src.train_models import train_and_evaluate

# Page configuration
st.set_page_config(
    page_title="AI in Retail & E-Commerce | Research Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Global Styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-paper {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .badge-exp {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #93C5FD;
    }
    .card-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .xai-box {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 0.9rem;
    }
    .guardrail-alert {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        color: #991B1B;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Ensure database is initialized
init_db()


@st.cache_data(ttl=60)
def fetch_cached_products():
    return load_products_df()


@st.cache_data(ttl=60)
def fetch_cached_customers():
    return load_customers_df()


@st.cache_data(ttl=60)
def fetch_cached_orders():
    return load_orders_df()


products_df = fetch_cached_products()
customers_df = fetch_cached_customers()
orders_df = fetch_cached_orders()

# Sidebar Navigation & Context
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/shopping-cart-loaded.png", width=64)
    st.markdown("### **Smart Retail AI**")
    st.caption("Academic Implementation of Prajapati et al. (2024)")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation Menu",
        [
            "🏠 Project Overview & Architecture",
            "🎯 Personalized Recommendation",
            "💬 Conversational AI Support",
            "📈 Adaptive Dynamic Pricing",
            "🔬 ML Model Evaluation",
            "📑 Research vs. Experimental Results",
            "🗄️ Database & Inventory Explorer"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### **System Status**")
    st.success("🟢 SQLite Database Connected")
    st.success("🟢 3 AI Service Engines Online")
    st.info("ℹ️ Using Clearly-Labelled Synthetic Demo Data")

    st.markdown("---")
    st.caption("Project: *Artificial Intelligence in Retail and E-commerce: Transforming Customer Experience*")


# -------------------------------------------------------------------------------------------------
# PAGE 1: PROJECT OVERVIEW & ARCHITECTURE
# -------------------------------------------------------------------------------------------------
if nav_choice == "🏠 Project Overview & Architecture":
    st.markdown('<div class="main-title">Artificial Intelligence in Retail & E-commerce</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transforming Customer Experience through Personalized Recommendations, Conversational Support, Dynamic Pricing & Hybrid AI Models</div>', unsafe_allow_html=True)

    # Top KPI Metrics Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Product Catalog", f"{len(products_df)} Items", "Across 6 Categories")
    with k2:
        st.metric("Active Customers", f"{len(customers_df)} Profiles", "4 Loyalty Tiers")
    with k3:
        st.metric("Trackable Orders", f"{len(orders_df)} Orders", "Live Statuses")
    with k4:
        st.metric("AI Core Modules", "3 / 3 Active", "Recs • Chat • Pricing")
    with k5:
        st.metric("Comparative ML", "4 Models", "LR • RF • XGB • Hybrid")

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.subheader("📌 Academic Research Framework")
        st.markdown("""
        This application represents the complete end-to-end implementation of the architecture proposed by 
        **R N Prajapati, Prashant Kumar, Aditya Prasad, Mayank Rai, Sunny Kumar, and Gaurav Kumar** 
        *(Chandigarh University, Mohali, India)*.

        ### **Core Research Scope & Objectives:**
        1. **Integrated AI Ecosystem**: Unifies recommendation, customer support, and dynamic pricing onto a single coordinated data platform (addressing Research Gap 1).
        2. **Fairness, Trust & Transparency**: Implements strict price surge caps, floor bounds, and essential goods protection to uphold consumer welfare and customer control (addressing Research Gap 2).
        3. **Hybrid AI Methodology**: Demonstrates why ensembling complementary machine learning models (combining linear boundaries, decision trees, and gradient boosting) improves predictive accuracy and recall for customer purchase conversion.
        """)

        st.info("""
        🔬 **Research Integrity Notice**:
        The dataset utilized in this prototype is synthetically generated according to the feature construction 
        methodology outlined in Section IV-A (Figure 2) of the paper. Paper-reported benchmarks (Table I) are 
        displayed side-by-side with our experimental measurements in the comparative analysis page.
        """)

    with col_right:
        st.subheader("🏛️ Four-Layer System Architecture")
        st.markdown("""
        ```
        ┌──────────────────────────────────────────────────────────┐
        │  Layer 1: Hardware & Data Acquisition                    │
        │  (Browsing, Searches, Clicks, POS, Cart, Purchases)     │
        └───────────────────────────┬──────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────┐
        │  Layer 2: Cloud & Database Storage (SQLite DBMS)         │
        │  (Products, Customers, Orders, Transactions, Logs)       │
        └───────────────────────────┬──────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────┐
        │  Layer 3: AI & Machine Learning Intelligence Engine      │
        │  ├── 1. Personalized Hybrid Recommender (TF-IDF + Collab)│
        │  ├── 2. Conversational AI Engine (Intent + DB Queries)   │
        │  ├── 3. Dynamic Pricing (Multi-Signal + Fairness Caps)   │
        │  └── 4. Hybrid Classifier (LR + RF + XGBoost Voting)     │
        └───────────────────────────┬──────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────┐
        │  Layer 4: User Interface & Analytics Dashboard           │
        │  (Streamlit Web Application & Interactive Visuals)       │
        └──────────────────────────────────────────────────────────┘
        ```
        """)

    st.markdown("---")
    st.subheader("🚀 Quick Feature Access")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown("#### 🎯 Recommendations")
        st.write("Hybrid content + collaborative recommendations with Explainable AI justifications.")
    with q2:
        st.markdown("#### 💬 AI Chatbot")
        st.write("Conversational customer support with real-time order tracking and product search.")
    with q3:
        st.markdown("#### 📈 Dynamic Pricing")
        st.write("Adaptive price optimization with demand sensitivity and fairness guardrails.")
    with q4:
        st.markdown("#### 🔬 Model Evaluation")
        st.write("Evaluation of Logistic Regression, Random Forest, XGBoost, and Proposed Hybrid.")


# -------------------------------------------------------------------------------------------------
# PAGE 2: PERSONALIZED PRODUCT RECOMMENDATIONS
# -------------------------------------------------------------------------------------------------
elif nav_choice == "🎯 Personalized Recommendation":
    st.markdown('<div class="main-title">Personalized Product Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hybrid Content-Based & Collaborative Filtering with Explainable AI (XAI) Justifications</div>', unsafe_allow_html=True)

    recommender = get_recommender()

    # Customer Profile Selector & Preference Inputs
    with st.expander("👤 Customer Profile & Simulation Settings", expanded=True):
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            customer_options = ["None (Guest Shopper)"] + [
                f"{row['customer_id']} — {row['name']} ({row['loyalty_tier']}, Prefers: {row['preferred_category']})"
                for _, row in customers_df.head(20).iterrows()
            ]
            selected_cust_str = st.selectbox("Select Customer Context", customer_options)
            if selected_cust_str != "None (Guest Shopper)":
                sel_cid = selected_cust_str.split(" — ")[0]
                cust_profile = recommender.get_customer_profile(sel_cid)
            else:
                sel_cid = None
                cust_profile = None

        with p_col2:
            categories_list = ["All Categories"] + sorted(products_df["category"].unique().tolist())
            default_cat_idx = 0
            if cust_profile and cust_profile.get("preferred_category") in categories_list:
                default_cat_idx = categories_list.index(cust_profile["preferred_category"])
            selected_cat = st.selectbox("Category Filter", categories_list, index=default_cat_idx)

        with p_col3:
            default_budget = float(cust_profile.get("budget", 8000.0)) if cust_profile else 8000.0
            budget_input = st.number_input(
                "Maximum Shopping Budget (₹)",
                min_value=500.0,
                max_value=100000.0,
                value=float(default_budget),
                step=500.0
            )

        # Advanced preferences & text search
        f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
        with f_col1:
            search_query = st.text_input("Semantic / Keyword Search", placeholder="e.g. wireless, ergonomic, waterproof, stainless steel")
        with f_col2:
            min_rating_input = st.slider("Minimum Customer Rating", 0.0, 5.0, 4.0, 0.1)
        with f_col3:
            in_stock_toggle = st.checkbox("In-Stock Items Only", value=True)

        # Optional Model Weights Tuning
        with st.expander("⚙️ Fine-Tune Hybrid Model Fusion Weights"):
            w_c1, w_c2, w_c3 = st.columns(3)
            with w_c1:
                w_content = st.slider("Content Feature Weight", 0.0, 1.0, 0.45, 0.05)
            with w_c2:
                w_collab = st.slider("Collaborative History Weight", 0.0, 1.0, 0.35, 0.05)
            with w_c3:
                w_rating = st.slider("Product Rating Weight", 0.0, 1.0, 0.20, 0.05)

    # Show active customer details if selected
    if cust_profile:
        st.markdown(
            f"**Active Shopper Profile**: `{cust_profile['name']}` | "
            f"**Loyalty Tier**: `{cust_profile['loyalty_tier']}` | "
            f"**Preferred Category**: `{cust_profile['preferred_category']}` | "
            f"**Recent Category Purchases**: {cust_profile.get('purchased_categories', 'None')}"
        )

    # Generate Recommendations
    target_category = None if selected_cat == "All Categories" else selected_cat

    recs = recommender.recommend(
        customer_id=sel_cid,
        category=target_category,
        max_budget=budget_input,
        min_rating=min_rating_input,
        search_query=search_query if search_query.strip() else None,
        in_stock_only=in_stock_toggle,
        top_k=6,
        w_content=w_content,
        w_collab=w_collab,
        w_rating=w_rating
    )

    st.markdown("---")
    st.subheader(f"✨ Top Recommended Products ({len(recs)} matches)")

    if not recs:
        st.warning("⚠️ No products found matching your current filter criteria. Try increasing the budget or lowering the rating threshold.")
    else:
        # Display recommendations in responsive grid cards
        for i in range(0, len(recs), 2):
            card_col1, card_col2 = st.columns(2)
            cols = [card_col1, card_col2]

            for j in range(2):
                idx = i + j
                if idx < len(recs):
                    item = recs[idx]
                    with cols[j]:
                        stock_pill = f"🟢 In Stock ({item['stock']} left)" if item['stock'] > 0 else "🔴 Out of Stock"
                        st.markdown(f"""
                        <div class="card-box">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:700; font-size:1.15rem; color:#1E3A8A;">#{item['rank']} {item['name']}</span>
                                <span class="badge-exp">{item['match_score_pct']}% Match</span>
                            </div>
                            <div style="margin-top:6px; color:#64748B; font-size:0.9rem;">
                                <b>{item['brand']}</b> • {item['category']} • ⭐ {item['rating']} ({item['reviews_count']} reviews) • {stock_pill}
                            </div>
                            <div style="margin-top:8px; font-size:0.92rem; color:#334155;">
                                {item['description']}
                            </div>
                            <div style="margin-top:10px; font-size:1.3rem; font-weight:700; color:#0F172A;">
                                ₹{item['price']:,.2f}
                            </div>
                            <div class="xai-box">
                                <b>💡 Explainable AI Justification:</b><br>
                                {"<br>".join(f"• {exp}" for exp in item['explanations'])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# -------------------------------------------------------------------------------------------------
# PAGE 3: CONVERSATIONAL AI CUSTOMER SUPPORT
# -------------------------------------------------------------------------------------------------
elif nav_choice == "💬 Conversational AI Support":
    st.markdown('<div class="main-title">Conversational AI Customer Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Local Natural Language Understanding (NLU) Prototype with Real-Time Database Order Tracking</div>', unsafe_allow_html=True)

    chatbot = get_chatbot()

    # Session State for Conversation History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! Welcome to Smart Retail AI Support.**\n\n"
                    "I am your automated AI shopping assistant. I can help you:\n"
                    "- 📦 **Track your orders** (e.g., *'Where is my order ORD-1002?'*)\n"
                    "- 🔍 **Search the product catalog** (e.g., *'Do you have wireless headphones?'*)\n"
                    "- 🔄 **Provide Return & Refund policies**\n"
                    "- 💳 **Answer Payment, EMI & Billing inquiries**\n"
                    "- ⛔ **Assist with Cancellations & Modifications**\n\n"
                    "How can I help you today?"
                )
            }
        ]

    # Quick Prompts / Intent Chips
    st.markdown("##### 💡 Suggested Questions (Click to test):")
    chip_col1, chip_col2, chip_col3, chip_col4, chip_col5 = st.columns(5)

    prompt_to_send = None
    with chip_col1:
        if st.button("📦 Track ORD-1002", use_container_width=True):
            prompt_to_send = "Where is my order ORD-1002?"
    with chip_col2:
        if st.button("🎧 Search Headphones", use_container_width=True):
            prompt_to_send = "Do you have wireless headphones?"
    with chip_col3:
        if st.button("🔄 30-Day Return Policy", use_container_width=True):
            prompt_to_send = "What is your return and refund policy?"
    with chip_col4:
        if st.button("💳 Payment Methods", use_container_width=True):
            prompt_to_send = "What payment options and UPI methods do you accept?"
    with chip_col5:
        if st.button("🤝 Talk to Human", use_container_width=True):
            prompt_to_send = "Can I speak with a customer care representative?"

    st.markdown("---")

    # Render Conversation History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat Input Box
    user_input = st.chat_input("Type your question or query here (e.g. Track order ORD-1005)...")
    active_prompt = prompt_to_send or user_input

    if active_prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Generate Bot Response
        result = chatbot.respond(active_prompt)
        bot_reply = result["reply"]

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    # Chat controls
    c_btn1, c_btn2, _ = st.columns([1, 1, 4])
    with c_btn1:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    st.caption("ℹ️ *Transparency Note: This module implements an explainable rule & intent NLU engine querying SQLite directly. It runs completely locally without requiring paid cloud LLM APIs.*")


# -------------------------------------------------------------------------------------------------
# PAGE 4: ADAPTIVE DYNAMIC PRICING & FAIRNESS SAFEGUARDS
# -------------------------------------------------------------------------------------------------
elif nav_choice == "📈 Adaptive Dynamic Pricing":
    st.markdown('<div class="main-title">Adaptive Dynamic Pricing Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Signal Market Optimization with Ethical Surge Limits and Essential Goods Protection</div>', unsafe_allow_html=True)

    pricing_engine = get_pricing_engine()

    col_sim, col_res = st.columns([1.1, 0.9])

    with col_sim:
        st.subheader("🎛️ Dynamic Pricing Input Signals")

        # Select product to price
        prod_options = [
            f"{row['product_id']} — {row['name']} (Base: ₹{row['base_price']:,.2f}, Stock: {row['stock']})"
            for _, row in products_df.iterrows()
        ]
        selected_prod_str = st.selectbox("Select Target Product from Catalog", prod_options)
        sel_pid = selected_prod_str.split(" — ")[0]
        sel_prod = products_df[products_df["product_id"] == sel_pid].iloc[0]

        base_price_val = float(sel_prod["base_price"])
        stock_val = int(sel_prod["stock"])
        rating_val = float(sel_prod["rating"])
        is_essential_val = bool(sel_prod["is_essential"])

        st.markdown(f"**Category**: `{sel_prod['category']}` | **Brand**: `{sel_prod['brand']}` | **Base Catalog Price**: `₹{base_price_val:,.2f}`")

        # Interactive Signal Sliders
        demand_signal = st.slider(
            "1. Real-Time Demand Index",
            0.0, 1.0, 0.75, 0.05,
            help="Computed from active views, search surge, cart additions, and peak shopping hours. 0.50 is neutral."
        )

        stock_units = st.number_input(
            "2. Current Available Inventory (Units)",
            min_value=0, max_value=200, value=stock_val, step=1,
            help="Inventory < 10 triggers low-stock scarcity premium; Inventory > 50 triggers clearance markdown."
        )

        competitor_price = st.number_input(
            "3. Competitor Market Price Benchmark (₹)",
            min_value=100.0, max_value=200000.0, value=float(base_price_val), step=100.0,
            help="Live scraped market price for comparable products."
        )

        with st.expander("🛡️ Ethical Fairness Guardrails & Policy"):
            is_essential_toggle = st.checkbox(
                "Mark as Essential Item (Healthcare / Basic Necessities)",
                value=is_essential_val,
                help="Strict consumer protection: Essential products are permanently exempted from surge pricing."
            )
            max_surge_cap = st.slider("Surge Ceiling Cap (+%)", 0.05, 0.50, 0.25, 0.05)
            max_discount_cap = st.slider("Floor Price Markdown Cap (-%)", 0.05, 0.40, 0.20, 0.05)

        pricing_engine.max_surge_pct = max_surge_cap
        pricing_engine.max_discount_pct = max_discount_cap

        # Calculate Price
        calc_result = pricing_engine.calculate_price(
            base_price=base_price_val,
            demand_signal=demand_signal,
            stock_units=stock_units,
            competitor_price=competitor_price,
            product_rating=rating_val,
            is_essential=is_essential_toggle,
            product_id=sel_pid,
            save_to_log=True
        )

    with col_res:
        st.subheader("💰 Suggested Price Output")

        # Metric Card
        st.markdown(f"""
        <div class="card-box">
            <div style="font-size:0.95rem; color:#64748B;">Suggested Adaptive Price</div>
            <div class="metric-value">₹{calc_result['suggested_price']:,.2f}</div>
            <div style="font-size:1.05rem; font-weight:600; color:{'#15803D' if calc_result['price_diff'] < 0 else '#B91C1C'};">
                {'+' if calc_result['price_diff'] > 0 else ''}₹{calc_result['price_diff']:,.2f} ({calc_result['price_diff_pct']:+.1f}% vs Catalog Base)
            </div>
        </div>
        """, unsafe_allow_html=True)

        if calc_result["bounded_flag"]:
            st.markdown(f"""
            <div class="guardrail-alert">
                🛡️ <b>Fairness Safeguard Triggered:</b><br>{calc_result['bounded_reason']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Breakdown of Price Signals:")
        st.write(calc_result["explanation"])

        # Signal Contribution Breakdown Chart
        signals_df = pd.DataFrame([
            {"Signal": "Demand Surge/Discount", "Impact (₹)": calc_result["demand_impact_inr"]},
            {"Signal": "Inventory Scarcity", "Impact (₹)": calc_result["scarcity_impact_inr"]},
            {"Signal": "Competitor Benchmark", "Impact (₹)": calc_result["comp_impact_inr"]},
            {"Signal": "Rating Quality", "Impact (₹)": calc_result["rating_impact_inr"]}
        ])
        fig_signals = px.bar(
            signals_df, x="Impact (₹)", y="Signal", orientation="h",
            color="Impact (₹)", color_continuous_scale="RdYlGn_r",
            title="Signal Impact Deconstruction (₹)"
        )
        fig_signals.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white")
        st.plotly_chart(fig_signals, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Price Sensitivity & Demand Elasticity Curve")
    st.caption("Simulating price adjustments across the full demand spectrum (0.0 to 1.0) for the selected item:")

    curve_df = pricing_engine.simulate_price_curve(base_price=base_price_val, stock_units=stock_units)
    fig_curve = px.line(
        curve_df, x="Demand Index", y="Suggested Price (₹)",
        title=f"Adaptive Price Curve vs. Demand Index — {sel_prod['name']}",
        markers=True,
        line_shape="spline"
    )
    fig_curve.add_hline(y=base_price_val, line_dash="dash", line_color="#94A3B8", annotation_text="Base Price")
    fig_curve.add_hline(y=base_price_val * (1 + max_surge_cap), line_dash="dot", line_color="#EF4444", annotation_text="Surge Ceiling")
    fig_curve.add_hline(y=base_price_val * (1 - max_discount_cap), line_dash="dot", line_color="#10B981", annotation_text="Floor Price")
    fig_curve.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_curve, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 5: MACHINE LEARNING MODEL EVALUATION DASHBOARD
# -------------------------------------------------------------------------------------------------
elif nav_choice == "🔬 ML Model Evaluation":
    st.markdown('<div class="main-title">Machine Learning Model Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparative Analysis of Logistic Regression, Random Forest, XGBoost, and the Proposed Hybrid AI Model</div>', unsafe_allow_html=True)

    exp_payload = load_experimental_metrics()

    if not exp_payload:
        st.warning("⚠️ No trained model metrics found yet. Click below to execute the training pipeline.")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models on synthetic retail dataset..."):
                exp_payload = train_and_evaluate(random_seed=42)
                st.success("Models trained successfully!")
                st.rerun()
    else:
        # Retrain button in header
        retrain_col, info_col = st.columns([1, 4])
        with retrain_col:
            if st.button("🔄 Retrain All Models"):
                with st.spinner("Retraining classifiers and recalculating metrics..."):
                    exp_payload = train_and_evaluate(random_seed=42)
                    st.success("Retraining complete!")
                    st.rerun()
        with info_col:
            st.caption(
                f"**Dataset**: `{exp_payload.get('dataset_name')}` ({exp_payload.get('total_samples')} total samples, {exp_payload.get('test_samples')} test samples) | "
                f"**Protocol**: `{exp_payload.get('split_ratio')}` | **Target**: `{exp_payload.get('target_variable')}`"
            )

        metrics_df = pd.DataFrame(exp_payload["metrics"])

        # Display Summary Metric KPI Cards
        st.markdown("#### 🏆 Experimental Performance Overview")
        m_cols = st.columns(len(metrics_df))
        for idx, row in metrics_df.iterrows():
            with m_cols[idx]:
                st.markdown(f"""
                <div class="card-box">
                    <span class="badge-exp">{row['model']}</span>
                    <div style="margin-top:8px;">
                        <span style="font-size:0.85rem; color:#64748B;">Accuracy</span>
                        <div class="metric-value">{row['accuracy']:.1f}%</div>
                    </div>
                    <div style="font-size:0.85rem; color:#475569; margin-top:4px;">
                        F1-Score: <b>{row['f1_score']:.1f}%</b><br>
                        Precision: <b>{row['precision']:.1f}%</b><br>
                        Recall: <b>{row['recall']:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Interactive Chart & Tabular View
        st.subheader("📊 Comparative Performance Across All Metrics")
        chart_col, radar_col = st.columns([1.2, 0.8])

        with chart_col:
            fig_grouped = plot_metrics_comparison_chart(
                metrics_df,
                title="Experimental Accuracy, Precision, Recall & F1-Score"
            )
            st.plotly_chart(fig_grouped, use_container_width=True)

        with radar_col:
            fig_radar = plot_radar_comparison(metrics_df, title="Multivariate Model Balance")
            st.plotly_chart(fig_radar, use_container_width=True)

        # Tabular View
        st.dataframe(
            metrics_df.style.format({
                "accuracy": "{:.2f}%",
                "precision": "{:.2f}%",
                "recall": "{:.2f}%",
                "f1_score": "{:.2f}%",
                "roc_auc": "{:.2f}%"
            }).highlight_max(subset=["accuracy", "precision", "recall", "f1_score"], color="#DCFCE7"),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Confusion Matrices Section
        st.subheader("🧩 Confusion Matrix Analysis")
        st.markdown("Analyze True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN):")

        cm_dict = load_confusion_matrices()
        if cm_dict:
            cm_model_choice = st.selectbox("Select Model Confusion Matrix", list(cm_dict.keys()))
            cm_col, stats_col = st.columns([1.1, 0.9])

            with cm_col:
                fig_cm = plot_confusion_matrix_heatmap(cm_model_choice, cm_dict[cm_model_choice])
                st.plotly_chart(fig_cm, use_container_width=True)

            with stats_col:
                cm = cm_dict[cm_model_choice]
                tot = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
                st.markdown(f"#### Matrix Breakdown for **{cm_model_choice}**")
                st.markdown(f"""
                - **True Negatives (TN)**: `{cm['tn']}` (Correctly identified non-purchasers)
                - **False Positives (FP)**: `{cm['fp']}` (Incorrectly predicted purchase — Type I error)
                - **False Negatives (FN)**: `{cm['fn']}` (Missed actual purchasers — Type II error)
                - **True Positives (TP)**: `{cm['tp']}` (Correctly predicted purchasers)
                - **Total Test Samples**: `{tot}`
                """)
                st.info("""
                💡 **Academic Insight**: In retail marketing and conversion prediction, minimizing **False Negatives (FN)** 
                is crucial to capture high-value conversion intent while avoiding intrusive marketing to uninterested visitors.
                """)

        # Feature Importance Section
        fi_dict = load_feature_importances()
        if fi_dict:
            st.markdown("---")
            st.subheader("🔍 Feature Importance & Predictive Retail Drivers")
            fi_models = list(fi_dict.keys())
            if fi_models:
                fi_choice = st.selectbox("Select Model for Feature Importances", fi_models)
                fig_fi = plot_feature_importance_chart(fi_choice, fi_dict[fi_choice])
                st.plotly_chart(fig_fi, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 6: RESEARCH PAPER VS. EXPERIMENTAL RESULTS COMPARATIVE ANALYSIS
# -------------------------------------------------------------------------------------------------
elif nav_choice == "📑 Research vs. Experimental Results":
    st.markdown('<div class="main-title">Research Paper vs. Experimental Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous Side-by-Side Comparison Between Prajapati et al. (Table I) and Independently Measured Metrics</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; padding:16px; margin-bottom:20px;">
        <h4 style="color:#B45309; margin-top:0;">🛡️ Research Integrity & Provenance Declaration</h4>
        <p style="color:#78350F; font-size:0.95rem; margin-bottom:0;">
        The published research paper (<b>Prajapati et al., 2024</b>) reports machine learning performance across a proprietary, 
        undisclosed retail dataset. In compliance with strict academic integrity standards:
        <br>
        1. <b>Paper-Reported Results</b> are documented separately as reported in Table I of the research paper.
        <br>
        2. <b>Our Experimental Results</b> are independently measured and calculated on the transparent synthetic demonstration dataset.
        <br>
        3. Neither results are fabricated, nor are the local models forced or hardcoded to artificially mimic the paper's percentages.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Comparison Table
    st.subheader("📋 Side-by-Side Performance Comparison")
    comp_df = build_paper_vs_experimental_comparison()

    st.dataframe(
        comp_df.style.format({
            "Paper Accuracy (%)": "{:.1f}%",
            "Our Accuracy (%)": "{:.2f}%",
            "Diff Acc (%)": "{:+.2f}%",
            "Paper F1 (%)": "{:.1f}%",
            "Our F1 (%)": "{:.2f}%",
            "Diff F1 (%)": "{:+.2f}%",
            "Paper Precision (%)": "{:.1f}%",
            "Our Precision (%)": "{:.2f}%",
            "Paper Recall (%)": "{:.1f}%",
            "Our Recall (%)": "{:.2f}%",
        }).background_gradient(subset=["Diff Acc (%)", "Diff F1 (%)"], cmap="coolwarm"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Comparative Visualizations
    st.subheader("📊 Visual Benchmark Comparison")
    c1, c2 = st.columns(2)

    with c1:
        # Paper Reported Bar Chart
        paper_df = pd.DataFrame(PAPER_REPORTED_RESULTS)
        fig_paper = px.bar(
            paper_df, x="model", y=["accuracy", "f1_score"], barmode="group",
            title="Paper-Reported Results (Table I — Prajapati et al.)",
            labels={"value": "Percentage (%)", "model": "Model", "variable": "Metric"},
            color_discrete_sequence=["#F59E0B", "#D97706"]
        )
        fig_paper.update_layout(yaxis=dict(range=[70, 102]), template="plotly_white", height=380)
        st.plotly_chart(fig_paper, use_container_width=True)

    with c2:
        # Experimental Run Bar Chart
        exp_payload = load_experimental_metrics()
        if exp_payload:
            exp_chart_df = pd.DataFrame(exp_payload["metrics"])
            fig_exp = px.bar(
                exp_chart_df, x="model", y=["accuracy", "f1_score"], barmode="group",
                title="Our Experimental Run (Synthetic Retail Dataset)",
                labels={"value": "Percentage (%)", "model": "Model", "variable": "Metric"},
                color_discrete_sequence=["#3B82F6", "#2563EB"]
            )
            fig_exp.update_layout(yaxis=dict(range=[50, 102]), template="plotly_white", height=380)
            st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("---")

    # In-Depth Research vs Implementation Audit
    st.subheader("📑 Systematic Implementation Audit")

    audit_data = [
        {
            "Dimension": "System Architecture",
            "Proposed in Paper": "4 Layers: Hardware/Data Acquisition, Cloud/DBMS, AI/ML Layer, UI Application (Fig. 1)",
            "Implemented in Prototype": "Complete 4-layer architecture with SQLite DBMS, modular AI engines, and Streamlit dashboard.",
            "Status": "✅ Fully Implemented"
        },
        {
            "Dimension": "Personalized Recommendation",
            "Proposed in Paper": "Hybrid Recommender combining Content-Based and Collaborative Behavioral relationships (Section IV-B)",
            "Implemented in Prototype": "TF-IDF text similarity + customer purchase history affinity + budget/stock filtering + XAI explanations.",
            "Status": "✅ Fully Implemented"
        },
        {
            "Dimension": "Conversational Customer Support",
            "Proposed in Paper": "Conversational AI to automate customer queries with communication quality and risk control (Section IV-B)",
            "Implemented in Prototype": "Local NLU intent engine handling order tracking, live catalog query, FAQs, and chat logging without paid APIs.",
            "Status": "✅ Fully Implemented"
        },
        {
            "Dimension": "Dynamic Pricing & Safeguards",
            "Proposed in Paper": "Adaptive pricing based on demand and market signals, with mitigations for privacy and fairness (Section IV-B & VII-B)",
            "Implemented in Prototype": "Multi-factor pricing (demand, scarcity, competitor, rating) with surge ceiling, floor price, and essential goods protection.",
            "Status": "✅ Fully Implemented"
        },
        {
            "Dimension": "Machine Learning Pipeline",
            "Proposed in Paper": "Comparison of Logistic Regression, Random Forest, XGBoost, and Proposed Hybrid AI Model (Table I)",
            "Implemented in Prototype": "Independent training of all 4 models, soft-voting ensemble, confusion matrices, and classification reports.",
            "Status": "✅ Fully Implemented"
        },
        {
            "Dimension": "Dataset Provenance",
            "Proposed in Paper": "Proprietary retail transactions and customer interactions (details unpublished in paper)",
            "Implemented in Prototype": "Transparently generated synthetic retail dataset matching Figure 2 feature construction protocol.",
            "Status": "ℹ️ Labelled Demonstration"
        }
    ]

    st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------------------------------------
# PAGE 7: DATABASE & INVENTORY EXPLORER
# -------------------------------------------------------------------------------------------------
elif nav_choice == "🗄️ Database & Inventory Explorer":
    st.markdown('<div class="main-title">Relational Database & Inventory Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect Live SQLite Tables: Products, Customer Profiles, Orders, and Dynamic Pricing Logs</div>', unsafe_allow_html=True)

    tab_prod, tab_cust, tab_orders, tab_logs = st.tabs([
        "📦 Products Catalog",
        "👥 Customer Profiles",
        "🚚 Trackable Orders",
        "📈 Dynamic Pricing Audit Logs"
    ])

    with tab_prod:
        st.subheader(f"Product Catalog ({len(products_df)} Items)")
        cat_filter = st.selectbox("Filter by Category", ["All"] + sorted(products_df["category"].unique().tolist()))
        df_show = products_df if cat_filter == "All" else products_df[products_df["category"] == cat_filter]
        st.dataframe(
            df_show[[
                "product_id", "name", "category", "brand", "base_price",
                "current_price", "rating", "reviews_count", "stock", "is_essential"
            ]],
            use_container_width=True,
            hide_index=True
        )

    with tab_cust:
        st.subheader(f"Registered Customer Profiles ({len(customers_df)} Shoppers)")
        tier_filter = st.selectbox("Filter by Loyalty Tier", ["All", "Bronze", "Silver", "Gold", "Platinum"])
        c_show = customers_df if tier_filter == "All" else customers_df[customers_df["loyalty_tier"] == tier_filter]
        st.dataframe(c_show, use_container_width=True, hide_index=True)

    with tab_orders:
        st.subheader(f"Customer Orders ({len(orders_df)} Trackable Orders)")
        st.dataframe(orders_df, use_container_width=True, hide_index=True)

    with tab_logs:
        st.subheader("Dynamic Pricing Real-Time Audit Log")
        pricing_logs_df = get_pricing_history(limit=50)
        if pricing_logs_df.empty:
            st.info("No pricing calculations logged yet. Test the Dynamic Pricing module to record events!")
        else:
            st.dataframe(pricing_logs_df, use_container_width=True, hide_index=True)
