"""Personalized Product Recommendation Engine.
Implements the hybrid recommendation architecture described in Section IV-B
of the research paper: combines Content-Based Filtering (product features,
descriptions, TF-IDF cosine similarity, budget fit) with Collaborative
Behavioral Signals (customer purchase history, category affinity, ratings).
Provides explainable AI (XAI) justifications for each recommendation.
"""
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import load_products_df, load_customers_df, get_db_connection


class HybridRecommendationEngine:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self._refresh_data()

    def _refresh_data(self):
        """Load catalog, customer profiles, and transaction logs."""
        self.products = load_products_df(self.db_path)
        self.customers = load_customers_df(self.db_path)

        # Build TF-IDF representation of product text attributes
        if not self.products.empty:
            self.products["combined_content"] = (
                self.products["category"] + " " +
                self.products["brand"] + " " +
                self.products["name"] + " " +
                self.products["description"] + " " +
                self.products["tags"].fillna("")
            )
            self.tfidf = TfidfVectorizer(stop_words="english", max_features=500)
            self.tfidf_matrix = self.tfidf.fit_transform(self.products["combined_content"])
        else:
            self.tfidf = None
            self.tfidf_matrix = None

    def get_customer_profile(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve existing customer profile, recent purchases, and category affinities."""
        if self.customers.empty or "customer_id" not in self.customers.columns:
            return None
        match = self.customers[self.customers["customer_id"] == customer_id]
        if match.empty:
            return None
        cust = match.iloc[0].to_dict()

        # Query past purchase history from transactions
        conn = get_db_connection(self.db_path)
        try:
            tx_df = pd.read_sql_query(
                "SELECT t.*, p.name as product_name, p.category FROM transactions t "
                "JOIN products p ON t.product_id = p.product_id "
                "WHERE t.customer_id = ? ORDER BY t.purchase_date DESC LIMIT 10",
                conn,
                params=(customer_id,)
            )
            cust["past_purchases"] = tx_df.to_dict(orient="records")
            cust["purchased_categories"] = tx_df["category"].value_counts().to_dict() if not tx_df.empty else {}
        finally:
            conn.close()

        return cust

    def recommend(
        self,
        customer_id: Optional[str] = None,
        category: Optional[str] = None,
        max_budget: Optional[float] = None,
        min_rating: float = 0.0,
        search_query: Optional[str] = None,
        in_stock_only: bool = True,
        top_k: int = 6,
        w_content: float = 0.45,
        w_collab: float = 0.35,
        w_rating: float = 0.20
    ) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations with full explainability.

        Weights:
        - w_content: Content similarity & attribute alignment
        - w_collab: Customer behavioral & historical affinity
        - w_rating: Quality score based on customer ratings
        """
        if self.products.empty:
            self._refresh_data()
            if self.products.empty:
                return []

        df = self.products.copy()

        # 1. Base eligibility filtering
        if in_stock_only:
            df = df[df["stock"] > 0]

        if min_rating > 0:
            df = df[df["rating"] >= min_rating]

        if df.empty:
            return []

        # Customer context (if valid customer_id provided)
        cust_profile = self.get_customer_profile(customer_id) if customer_id else None
        target_category = category or (cust_profile.get("preferred_category") if cust_profile else None)
        target_budget = max_budget or (float(cust_profile.get("budget", 50000)) if cust_profile else None)
        past_cats = cust_profile.get("purchased_categories", {}) if cust_profile else {}

        # 2. Content-Based Score Calculation
        content_scores = np.zeros(len(df))

        if search_query and self.tfidf is not None:
            # Query vector similarity
            query_vec = self.tfidf.transform([search_query])
            sub_indices = df.index.values
            sims = cosine_similarity(query_vec, self.tfidf_matrix[sub_indices]).flatten()
            content_scores += sims * 0.70

        # Category alignment
        if target_category and target_category != "All Categories":
            cat_match = (df["category"] == target_category).astype(float).values
            content_scores += cat_match * 0.50
        else:
            content_scores += 0.25  # baseline

        # Budget fit scoring (smooth penalization if exceeding budget)
        if target_budget and target_budget > 0:
            price_ratios = df["current_price"].values / target_budget
            # 1.0 if inside budget, smoothly drops off if above budget
            budget_scores = np.where(price_ratios <= 1.0, 1.0 - (price_ratios * 0.3), np.maximum(0.0, 1.0 - (price_ratios - 1.0) * 1.5))
            content_scores += budget_scores * 0.40

        # Normalize content score to [0, 1]
        c_max = np.max(content_scores) if np.max(content_scores) > 0 else 1.0
        norm_content = content_scores / c_max

        # 3. Collaborative & Behavioral Affinity Score
        collab_scores = np.zeros(len(df))
        for idx, (_, row) in enumerate(df.iterrows()):
            cat = row["category"]
            cat_purchases = past_cats.get(cat, 0)
            # Preference affinity
            pref_boost = 0.4 if (cust_profile and cust_profile.get("preferred_category") == cat) else 0.0
            # History affinity
            hist_boost = min(0.4, cat_purchases * 0.15)
            # Reviews count signal (social proof / popularity proxy)
            pop_score = min(0.2, (row["reviews_count"] / 1000.0) * 0.2)
            collab_scores[idx] = pref_boost + hist_boost + pop_score

        collab_max = np.max(collab_scores) if np.max(collab_scores) > 0 else 1.0
        norm_collab = collab_scores / collab_max if np.max(collab_scores) > 0 else np.full(len(df), 0.3)

        # 4. Rating & Quality Score
        norm_rating = (df["rating"].values - 3.0) / 2.0  # normalize 3.0-5.0 scale to 0.0-1.0
        norm_rating = np.clip(norm_rating, 0.0, 1.0)

        # 5. Hybrid Weighted Score
        hybrid_scores = (
            w_content * norm_content +
            w_collab * norm_collab +
            w_rating * norm_rating
        )

        df["content_score"] = np.round(norm_content, 3)
        df["collab_score"] = np.round(norm_collab, 3)
        df["rating_score"] = np.round(norm_rating, 3)
        df["hybrid_match_pct"] = np.round(hybrid_scores * 100, 1)

        # Sort descending by hybrid match score
        df = df.sort_values(by=["hybrid_match_pct", "rating", "reviews_count"], ascending=[False, False, False])

        # Filter strictly by budget if user set hard constraint and results exist
        if target_budget:
            within_budget = df[df["current_price"] <= target_budget]
            if len(within_budget) >= min(3, top_k):
                df = within_budget

        results = []
        for rank, (_, row) in enumerate(df.head(top_k).iterrows(), start=1):
            explanations = []

            # Generate XAI explanation points
            if target_category and row["category"] == target_category:
                explanations.append(f"Category Match: Perfectly matches preferred category '{row['category']}'.")
            elif row["category"] in past_cats:
                explanations.append(f"Behavioral Affinity: Aligns with your past purchases in '{row['category']}'.")

            if target_budget:
                if row["current_price"] <= target_budget:
                    savings = target_budget - row["current_price"]
                    explanations.append(f"Budget Fit: ₹{row['current_price']:,.2f} fits within your ₹{target_budget:,.2f} budget (₹{savings:,.2f} headroom).")
                else:
                    explanations.append(f"Budget Stretch: Premium option slightly above target budget.")

            if row["rating"] >= 4.5:
                explanations.append(f"Customer Favorite: High user satisfaction rating of {row['rating']}★ ({row['reviews_count']} verified reviews).")

            if row["stock"] < 10:
                explanations.append(f"Limited Inventory: High demand item with only {row['stock']} units remaining.")
            else:
                explanations.append(f"Availability: In stock and ready to ship ({row['stock']} units).")

            if cust_profile and cust_profile.get("loyalty_tier") in ["Gold", "Platinum"]:
                explanations.append(f"Loyalty Perk: Eligible for {cust_profile['loyalty_tier']} tier expedited priority dispatch.")

            results.append({
                "rank": rank,
                "product_id": row["product_id"],
                "name": row["name"],
                "category": row["category"],
                "brand": row["brand"],
                "price": float(row["current_price"]),
                "base_price": float(row["base_price"]),
                "rating": float(row["rating"]),
                "reviews_count": int(row["reviews_count"]),
                "stock": int(row["stock"]),
                "description": row["description"],
                "tags": str(row.get("tags", "")),
                "match_score_pct": float(row["hybrid_match_pct"]),
                "explanations": explanations
            })

        return results


# Module singleton instance
_engine_instance: Optional[HybridRecommendationEngine] = None

def get_recommender(db_path: Optional[Path] = None) -> HybridRecommendationEngine:
    global _engine_instance
    if _engine_instance is None or db_path is not None:
        _engine_instance = HybridRecommendationEngine(db_path)
    return _engine_instance


def recommend_products(
    category: Optional[str] = None,
    budget: float = 50000.0,
    customer_id: Optional[str] = None,
    limit: int = 6,
    min_rating: float = 0.0,
    search_query: Optional[str] = None
) -> pd.DataFrame:
    """Convenience helper returning recommendations as a DataFrame."""
    engine = get_recommender()
    recs = engine.recommend(
        customer_id=customer_id,
        category=category,
        max_budget=budget,
        min_rating=min_rating,
        search_query=search_query,
        top_k=limit
    )
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs)
