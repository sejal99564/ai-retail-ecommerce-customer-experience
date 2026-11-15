"""Database management module for AI Retail and E-commerce system.
Manages SQLite database schema, connections, and transactional queries
for products, customers, transactions, orders, interactions, and logs.
"""
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "retail_ecommerce.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create and return a connection to the SQLite database."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize all relational database tables according to paper architecture."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Products Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            brand TEXT NOT NULL,
            base_price REAL NOT NULL,
            current_price REAL NOT NULL,
            rating REAL NOT NULL,
            reviews_count INTEGER NOT NULL,
            stock INTEGER NOT NULL,
            description TEXT NOT NULL,
            tags TEXT,
            is_essential INTEGER DEFAULT 0
        )
    """)

    # 2. Customers Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            preferred_category TEXT NOT NULL,
            budget REAL NOT NULL,
            loyalty_tier TEXT NOT NULL,
            join_date TEXT NOT NULL
        )
    """)

    # 3. Orders Table (for Conversational AI Order Tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL,
            tracking_number TEXT NOT NULL,
            estimated_delivery TEXT NOT NULL,
            shipping_address TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)

    # 4. Transactions Table (for Historical Analytics & Collaborative Signals)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            purchase_date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            total_amount REAL NOT NULL,
            payment_method TEXT NOT NULL,
            satisfaction_rating REAL,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)

    # 5. Customer Interactions Table (Browsing, Clicks, Cart additions)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            interaction_type TEXT NOT NULL,
            session_id TEXT NOT NULL,
            duration_seconds INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)

    # 6. Chat History Table (Preserves Multi-turn Conversation & Intent Logs)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            customer_id TEXT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            intent TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 7. Dynamic Pricing Audit Logs Table (Transparent price adaptation logs)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pricing_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT NOT NULL,
            base_price REAL NOT NULL,
            suggested_price REAL NOT NULL,
            demand_factor REAL NOT NULL,
            scarcity_factor REAL NOT NULL,
            competitor_factor REAL NOT NULL,
            bounded_flag INTEGER DEFAULT 0,
            explanation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    """)

    conn.commit()
    conn.close()


def load_products_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load all products as a pandas DataFrame."""
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM products ORDER BY category, name", conn)
        return df
    finally:
        conn.close()


def load_customers_df(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load all customers as a pandas DataFrame."""
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM customers ORDER BY customer_id", conn)
        return df
    finally:
        conn.close()


def load_orders_df(customer_id: Optional[str] = None, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load orders optionally filtered by customer_id."""
    conn = get_db_connection(db_path)
    try:
        if customer_id:
            df = pd.read_sql_query(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date DESC",
                conn,
                params=(customer_id,)
            )
        else:
            df = pd.read_sql_query("SELECT * FROM orders ORDER BY order_date DESC", conn)
        return df
    finally:
        conn.close()


def get_order_by_id(order_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Look up a specific order by order_id (case-insensitive)."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE UPPER(order_id) = UPPER(?)", (order_id.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def search_products(query: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Search products matching name, category, or description."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        clean_q = "".join(c for c in query.strip() if c.isalnum() or c.isspace()).strip()
        search_pattern = f"%{clean_q}%"
        cursor.execute("""
            SELECT * FROM products 
            WHERE name LIKE ? OR category LIKE ? OR description LIKE ? OR tags LIKE ?
            ORDER BY rating DESC LIMIT 10
        """, (search_pattern, search_pattern, search_pattern, search_pattern))
        rows = cursor.fetchall()

        if not rows and " " in clean_q:
            # Fallback to token search if full phrase didn't match
            tokens = [t for t in clean_q.split() if len(t) > 2]
            if tokens:
                clauses = " OR ".join(["name LIKE ? OR description LIKE ? OR tags LIKE ?"] * len(tokens))
                params = []
                for t in tokens:
                    pat = f"%{t}%"
                    params.extend([pat, pat, pat])
                cursor.execute(f"SELECT * FROM products WHERE {clauses} ORDER BY rating DESC LIMIT 10", params)
                rows = cursor.fetchall()

        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_chat_message(session_id: str, role: str, message: str, intent: Optional[str] = None,
                     customer_id: Optional[str] = None, db_path: Optional[Path] = None) -> None:
    """Log conversational message into the chat_logs table."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_logs (session_id, customer_id, role, message, intent)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, customer_id, role, message, intent))
        conn.commit()
    finally:
        conn.close()


def log_pricing_event(product_id: str, base_price: float, suggested_price: float,
                      demand_factor: float, scarcity_factor: float, competitor_factor: float,
                      bounded_flag: int, explanation: str, db_path: Optional[Path] = None) -> None:
    """Log dynamic pricing calculation into pricing_logs table."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pricing_logs 
            (product_id, base_price, suggested_price, demand_factor, scarcity_factor, 
             competitor_factor, bounded_flag, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (product_id, base_price, suggested_price, demand_factor, scarcity_factor,
              competitor_factor, bounded_flag, explanation))
        conn.commit()
    finally:
        conn.close()


def get_pricing_history(limit: int = 50, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Get latest dynamic pricing history records."""
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT p.*, pr.name as product_name FROM pricing_logs p "
            "JOIN products pr ON p.product_id = pr.product_id "
            "ORDER BY p.timestamp DESC LIMIT ?",
            conn,
            params=(limit,)
        )
        return df
    finally:
        conn.close()
