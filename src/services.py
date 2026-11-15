"""Services facade module.
Provides a unified, backward-compatible API interface while delegating
to the modularized subsystem components:
- src.database (SQLite data layer)
- src.recommender (Hybrid recommendation engine)
- src.chatbot (Conversational AI customer support)
- src.dynamic_pricing (Adaptive pricing engine)
- src.evaluation (ML metrics and visualization)
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import load_products_df, load_customers_df, load_orders_df, get_order_by_id, init_db
from src.recommender import get_recommender, recommend_products
from src.chatbot import get_chatbot, chatbot_reply
from src.dynamic_pricing import get_pricing_engine, calculate_demo_price
from src.evaluation import PAPER_REPORTED_RESULTS, load_experimental_metrics, load_confusion_matrices

# Ensure SQLite DB is initialized
init_db()

def load_products() -> pd.DataFrame:
    """Load products catalog dataframe."""
    return load_products_df()

__all__ = [
    "load_products",
    "recommend_products",
    "chatbot_reply",
    "calculate_demo_price",
    "get_recommender",
    "get_chatbot",
    "get_pricing_engine",
    "PAPER_REPORTED_RESULTS",
    "load_experimental_metrics",
    "load_confusion_matrices"
]
