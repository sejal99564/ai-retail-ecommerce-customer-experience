"""Comprehensive automated test suite for AI Retail & E-commerce system.
Tests all core subsystems:
1. Database operations & relational queries
2. Personalized Hybrid Recommendation Engine & Explanations
3. Conversational AI Customer Support & Order Tracking
4. Adaptive Dynamic Pricing & Fairness Guardrails
5. Machine Learning Models, Training Pipeline & Metrics Loading
"""
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import (
    init_db, get_db_connection, load_products_df, load_customers_df,
    load_orders_df, get_order_by_id, search_products, log_chat_message,
    log_pricing_event, get_pricing_history
)
from src.recommender import get_recommender, recommend_products
from src.chatbot import get_chatbot, chatbot_reply
from src.dynamic_pricing import get_pricing_engine, calculate_demo_price
from src.evaluation import (
    PAPER_REPORTED_RESULTS, load_experimental_metrics,
    load_confusion_matrices, build_paper_vs_experimental_comparison
)


class TestDatabaseLayer(unittest.TestCase):
    """Test SQLite database initialization and querying."""

    def setUp(self):
        init_db()

    def test_products_table_not_empty(self):
        df = load_products_df()
        self.assertFalse(df.empty, "Products table should contain records.")
        self.assertIn("product_id", df.columns)
        self.assertIn("category", df.columns)
        self.assertIn("base_price", df.columns)
        self.assertIn("stock", df.columns)

    def test_customers_table_not_empty(self):
        df = load_customers_df()
        self.assertFalse(df.empty, "Customers table should contain records.")
        self.assertIn("customer_id", df.columns)
        self.assertIn("loyalty_tier", df.columns)

    def test_orders_query(self):
        orders = load_orders_df()
        self.assertFalse(orders.empty, "Orders table should contain trackable records.")
        sample_order = orders.iloc[0]
        oid = sample_order["order_id"]
        fetched = get_order_by_id(oid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["order_id"], oid)
        self.assertIn("status", fetched)

    def test_product_search(self):
        results = search_products("Headphones")
        self.assertTrue(len(results) > 0, "Search for 'Headphones' should return matches.")
        self.assertIn("product_id", results[0])


class TestRecommendationEngine(unittest.TestCase):
    """Test Hybrid Recommender functionality, budget enforcement, and XAI explanations."""

    def setUp(self):
        self.engine = get_recommender()

    def test_recommendation_returns_results(self):
        recs = self.engine.recommend(category="Electronics", max_budget=10000.0, top_k=5)
        self.assertTrue(len(recs) > 0, "Recommender should return products.")
        self.assertLessEqual(len(recs), 5)

    def test_budget_constraint_filtering(self):
        max_budget = 2500.0
        recs = self.engine.recommend(category="Fashion", max_budget=max_budget, top_k=4)
        for item in recs:
            self.assertLessEqual(item["price"], max_budget, "Item price must be within budget.")

    def test_explanation_generation(self):
        recs = self.engine.recommend(category="Home & Kitchen", max_budget=8000.0, top_k=3)
        self.assertTrue(len(recs) > 0)
        for item in recs:
            self.assertIn("explanations", item)
            self.assertTrue(len(item["explanations"]) >= 1, "Each product must include XAI explanation points.")

    def test_customer_context_recommendation(self):
        cust_df = load_customers_df()
        if not cust_df.empty:
            cid = cust_df.iloc[0]["customer_id"]
            recs = self.engine.recommend(customer_id=cid, top_k=4)
            self.assertTrue(len(recs) > 0)


class TestConversationalAI(unittest.TestCase):
    """Test Chatbot intent parsing, order tracking, and FAQ resolution."""

    def setUp(self):
        self.bot = get_chatbot()

    def test_order_status_tracking(self):
        orders = load_orders_df()
        if not orders.empty:
            oid = orders.iloc[0]["order_id"]
            res = self.bot.respond(f"Where is my order {oid}?")
            self.assertIn(oid, res["reply"])
            self.assertEqual(res["intent"], "order_status_success")

    def test_order_not_found(self):
        res = self.bot.respond("Check status of ORD-999999")
        self.assertIn("Order Not Found", res["reply"])
        self.assertEqual(res["intent"], "order_not_found")

    def test_shipping_faq_intent(self):
        res = self.bot.respond("What are your shipping and delivery options?")
        self.assertIn("Shipping & Delivery Policy", res["reply"])
        self.assertEqual(res["intent"], "faq_shipping")

    def test_returns_faq_intent(self):
        res = self.bot.respond("How can I return an item and get a refund?")
        self.assertIn("Returns & Refund Policy", res["reply"])
        self.assertEqual(res["intent"], "faq_returns")

    def test_payment_faq_intent(self):
        res = self.bot.respond("Do you accept UPI and Credit Cards?")
        self.assertIn("Payment & Security Options", res["reply"])
        self.assertEqual(res["intent"], "faq_payment")

    def test_product_inquiry(self):
        res = self.bot.respond("Do you have wireless headphones?")
        self.assertIn("Catalog Matches", res["reply"])
        self.assertEqual(res["intent"], "product_search")


class TestDynamicPricingEngine(unittest.TestCase):
    """Test Dynamic Pricing mathematical signals, fairness guardrails, and bounds."""

    def setUp(self):
        self.engine = get_pricing_engine()

    def test_high_demand_price_surge(self):
        base_price = 2000.0
        # High demand (0.95), low stock (5)
        res = self.engine.calculate_price(base_price=base_price, demand_signal=0.95, stock_units=5)
        self.assertGreater(res["suggested_price"], base_price, "High demand and low stock should increase price.")
        self.assertIn("explanation", res)

    def test_low_demand_markdown(self):
        base_price = 2000.0
        # Low demand (0.10), ample stock (80)
        res = self.engine.calculate_price(base_price=base_price, demand_signal=0.10, stock_units=80)
        self.assertLess(res["suggested_price"], base_price, "Low demand and overstock should trigger discount.")

    def test_surge_ceiling_guardrail(self):
        base_price = 1000.0
        # Extreme surge inputs: demand 1.0, stock 1, high competitor price
        res = self.engine.calculate_price(
            base_price=base_price, demand_signal=1.0, stock_units=1,
            competitor_price=3000.0, product_rating=5.0
        )
        max_allowed = base_price * (1.0 + self.engine.max_surge_pct)
        self.assertLessEqual(res["suggested_price"], max_allowed + 0.01, "Price must not exceed max surge ceiling.")

    def test_floor_price_guardrail(self):
        base_price = 1000.0
        # Extreme discount inputs: demand 0.0, stock 200, low competitor price
        res = self.engine.calculate_price(
            base_price=base_price, demand_signal=0.0, stock_units=200,
            competitor_price=200.0, product_rating=1.0
        )
        min_allowed = base_price * (1.0 - self.engine.max_discount_pct)
        self.assertGreaterEqual(res["suggested_price"], min_allowed - 0.01, "Price must not breach floor minimum.")

    def test_essential_goods_protection(self):
        base_price = 1000.0
        # High demand on essential product
        res = self.engine.calculate_price(
            base_price=base_price, demand_signal=0.99, stock_units=2,
            is_essential=True
        )
        self.assertLessEqual(res["suggested_price"], base_price, "Essential items must NEVER surge above base price.")
        self.assertEqual(res["bounded_flag"], 1)


class TestMachineLearningEvaluation(unittest.TestCase):
    """Test model metrics loading, confusion matrix structure, and comparison format."""

    def test_paper_reported_results_present(self):
        self.assertEqual(len(PAPER_REPORTED_RESULTS), 4)
        models = [p["model"] for p in PAPER_REPORTED_RESULTS]
        self.assertIn("Logistic Regression", models)
        self.assertIn("Random Forest", models)
        self.assertIn("XGBoost", models)
        self.assertIn("Proposed Hybrid AI Model", models)

    def test_experimental_metrics_loaded(self):
        payload = load_experimental_metrics()
        self.assertIsNotNone(payload, "Experimental metrics file should exist.")
        self.assertIn("metrics", payload)
        self.assertTrue(len(payload["metrics"]) >= 4)
        for m in payload["metrics"]:
            self.assertIn("accuracy", m)
            self.assertIn("precision", m)
            self.assertIn("recall", m)
            self.assertIn("f1_score", m)

    def test_confusion_matrices_loaded(self):
        cms = load_confusion_matrices()
        self.assertIsNotNone(cms)
        for m_name, cm in cms.items():
            self.assertIn("tn", cm)
            self.assertIn("fp", cm)
            self.assertIn("fn", cm)
            self.assertIn("tp", cm)

    def test_paper_vs_experimental_comparison_table(self):
        df = build_paper_vs_experimental_comparison()
        self.assertFalse(df.empty)
        self.assertIn("Model", df.columns)
        self.assertIn("Paper Accuracy (%)", df.columns)
        self.assertIn("Our Accuracy (%)", df.columns)


if __name__ == "__main__":
    unittest.main()
