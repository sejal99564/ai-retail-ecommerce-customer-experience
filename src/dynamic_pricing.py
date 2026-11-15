"""Adaptive Dynamic Pricing Module.
Implements the multi-factor pricing engine described in Section IV-B & VII-B
of the research paper. Evaluates real-time demand signals, stock availability,
competitor market benchmarks, and product ratings while strictly enforcing
fairness safeguards, essential goods protection, and surge limits.
"""
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import log_pricing_event, get_pricing_history, load_products_df


class DynamicPricingEngine:
    """
    Transparent, explainable multi-signal dynamic pricing engine with ethical fairness bounds.
    """

    def __init__(
        self,
        max_surge_pct: float = 0.25,     # Maximum allowable price increase (+25%)
        max_discount_pct: float = 0.20,  # Maximum allowable markdown (-20%)
        demand_elasticity: float = 0.22, # Sensitivity to consumer demand index
        scarcity_elasticity: float = 0.12,# Sensitivity to low stock
        competitor_elasticity: float = 0.15, # Sensitivity to competitor prices
        db_path: Optional[Path] = None
    ):
        self.max_surge_pct = max_surge_pct
        self.max_discount_pct = max_discount_pct
        self.demand_elasticity = demand_elasticity
        self.scarcity_elasticity = scarcity_elasticity
        self.competitor_elasticity = competitor_elasticity
        self.db_path = db_path

    def calculate_price(
        self,
        base_price: float,
        demand_signal: float,             # Normalized index: 0.0 (low) to 1.0 (high)
        stock_units: int,                 # Current available inventory count
        competitor_price: Optional[float] = None, # Market competitor price
        product_rating: float = 4.2,      # Product rating (1.0 to 5.0)
        is_essential: bool = False,       # Essential item protection flag
        product_id: Optional[str] = None,
        save_to_log: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate adaptive price and generate structured explanation and audit trail.
        """
        base = max(1.0, float(base_price))
        demand = np.clip(float(demand_signal), 0.0, 1.0)
        stock = max(0, int(stock_units))
        rating = np.clip(float(product_rating), 1.0, 5.0)

        # 1. Demand Component: Neutral at 0.50
        # delta_demand > 0 if demand > 0.5; < 0 if demand < 0.5
        delta_demand_pct = (demand - 0.50) * self.demand_elasticity

        # 2. Scarcity & Inventory Component:
        # High scarcity if stock < 10 units; markdown if overstocked (> 50 units)
        if stock == 0:
            delta_scarcity_pct = 0.0  # Cannot sell out of stock item
        elif stock < 10:
            # Low stock urgency factor
            urgency_ratio = (10 - stock) / 10.0
            delta_scarcity_pct = urgency_ratio * self.scarcity_elasticity
        elif stock > 50:
            # Overstock clearance discount
            excess_ratio = min(1.0, (stock - 50) / 100.0)
            delta_scarcity_pct = -1.0 * excess_ratio * 0.06
        else:
            delta_scarcity_pct = 0.0

        # 3. Competitor Market Signal Component
        if competitor_price is not None and competitor_price > 0:
            comp = float(competitor_price)
            comp_ratio = comp / base
            # If competitor is higher, room to increase; if lower, competitive discount
            delta_comp_pct = np.clip((comp_ratio - 1.0) * self.competitor_elasticity, -0.10, 0.10)
        else:
            comp = base
            delta_comp_pct = 0.0

        # 4. Rating & Quality Signal
        # Premium for 4.5+ star items; discount incentive for < 3.8
        delta_rating_pct = (rating - 4.2) * 0.03

        # Aggregate raw percentage adjustment
        total_raw_delta_pct = delta_demand_pct + delta_scarcity_pct + delta_comp_pct + delta_rating_pct
        raw_suggested = base * (1.0 + total_raw_delta_pct)

        # 5. Fairness Safeguards & Boundary Enforcement
        min_allowed = base * (1.0 - self.max_discount_pct)
        max_allowed = base * (1.0 + self.max_surge_pct)

        bounded_reason = "Unconstrained"
        bounded_flag = 0

        # Essential Goods Safeguard: No surge permitted on life-essential products!
        if is_essential and raw_suggested > base:
            suggested_price = base
            bounded_flag = 1
            bounded_reason = "Essential Goods Policy: Price surge strictly prohibited on basic health/essential items."
        elif raw_suggested > max_allowed:
            suggested_price = max_allowed
            bounded_flag = 1
            bounded_reason = f"Surge Ceiling Applied: Price increase capped at +{self.max_surge_pct*100:.0f}% to prevent price gouging."
        elif raw_suggested < min_allowed:
            suggested_price = min_allowed
            bounded_flag = 1
            bounded_reason = f"Floor Price Safeguard: Maximum discount capped at -{self.max_discount_pct*100:.0f}% to protect supplier margins."
        else:
            suggested_price = raw_suggested

        suggested_price = round(max(0.0, suggested_price), 2)
        total_effective_diff = suggested_price - base
        total_effective_pct = (total_effective_diff / base) * 100.0

        # Construct explainable factors
        demand_impact_inr = base * delta_demand_pct
        scarcity_impact_inr = base * delta_scarcity_pct
        comp_impact_inr = base * delta_comp_pct
        rating_impact_inr = base * delta_rating_pct

        explanation_lines = [
            f"• Demand Index ({demand:.2f}): {'Surge' if demand_impact_inr >= 0 else 'Discount'} of ₹{abs(demand_impact_inr):,.2f} ({delta_demand_pct*100:+.1f}%) based on active browsing interest.",
            f"• Inventory Level ({stock} units): {'Scarcity premium' if scarcity_impact_inr > 0 else ('Clearance incentive' if scarcity_impact_inr < 0 else 'Balanced stock')} of ₹{abs(scarcity_impact_inr):,.2f} ({delta_scarcity_pct*100:+.1f}%).",
            f"• Market Benchmark (₹{comp:,.2f}): Alignment adjustment of ₹{abs(comp_impact_inr):,.2f} ({delta_comp_pct*100:+.1f}%).",
            f"• Customer Rating ({rating}★): Quality adjustment of ₹{abs(rating_impact_inr):,.2f} ({delta_rating_pct*100:+.1f}%)."
        ]
        if bounded_flag:
            explanation_lines.append(f"🛡️ **Fairness Guardrail**: {bounded_reason}")

        explanation_text = "\n".join(explanation_lines)

        # Log event if requested
        if save_to_log and product_id:
            log_pricing_event(
                product_id=product_id,
                base_price=base,
                suggested_price=suggested_price,
                demand_factor=round(demand, 2),
                scarcity_factor=round(delta_scarcity_pct, 4),
                competitor_factor=round(delta_comp_pct, 4),
                bounded_flag=bounded_flag,
                explanation=explanation_text,
                db_path=self.db_path
            )

        return {
            "base_price": base,
            "suggested_price": suggested_price,
            "price_diff": round(total_effective_diff, 2),
            "price_diff_pct": round(total_effective_pct, 2),
            "demand_signal": demand,
            "stock_units": stock,
            "competitor_price": comp,
            "product_rating": rating,
            "is_essential": is_essential,
            "demand_impact_inr": round(demand_impact_inr, 2),
            "scarcity_impact_inr": round(scarcity_impact_inr, 2),
            "comp_impact_inr": round(comp_impact_inr, 2),
            "rating_impact_inr": round(rating_impact_inr, 2),
            "bounded_flag": bounded_flag,
            "bounded_reason": bounded_reason,
            "explanation": explanation_text
        }

    def simulate_price_curve(self, base_price: float, stock_units: int = 25) -> pd.DataFrame:
        """Simulate price response across demand range from 0.0 to 1.0."""
        demand_steps = np.linspace(0.0, 1.0, 21)
        sim_records = []
        for d in demand_steps:
            res = self.calculate_price(
                base_price=base_price,
                demand_signal=d,
                stock_units=stock_units,
                save_to_log=False
            )
            sim_records.append({
                "Demand Index": round(d, 2),
                "Suggested Price (₹)": res["suggested_price"],
                "Base Price (₹)": res["base_price"],
                "Price Difference (₹)": res["price_diff"],
                "Change (%)": res["price_diff_pct"]
            })
        return pd.DataFrame(sim_records)


# Global singleton instance
_pricing_engine: Optional[DynamicPricingEngine] = None

def get_pricing_engine(db_path: Optional[Path] = None) -> DynamicPricingEngine:
    global _pricing_engine
    if _pricing_engine is None or db_path is not None:
        _pricing_engine = DynamicPricingEngine(db_path=db_path)
    return _pricing_engine


def calculate_demo_price(base_price: float, demand: float, stock: int) -> float:
    """Starter compatibility helper."""
    engine = get_pricing_engine()
    res = engine.calculate_price(base_price=base_price, demand_signal=demand, stock_units=stock)
    return res["suggested_price"]
