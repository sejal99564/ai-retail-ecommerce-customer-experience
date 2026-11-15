"""Model Evaluation and Visualization Module.
Provides analytics routines, Plotly chart builders, confusion matrix heatmaps,
and comparison engines between Paper-Reported Metrics (Prajapati et al.) and
Independently Measured Experimental Metrics.
"""
from pathlib import Path
import json
import sys
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"


# Benchmark metrics reported in Table I of the research paper (Prajapati et al.)
PAPER_REPORTED_RESULTS = [
    {"model": "Logistic Regression", "accuracy": 89.4, "precision": 88.7, "recall": 87.9, "f1_score": 88.3},
    {"model": "Random Forest", "accuracy": 93.1, "precision": 92.6, "recall": 92.2, "f1_score": 92.4},
    {"model": "XGBoost", "accuracy": 95.0, "precision": 94.5, "recall": 94.1, "f1_score": 94.3},
    {"model": "Proposed Hybrid AI Model", "accuracy": 96.7, "precision": 96.2, "recall": 95.9, "f1_score": 96.0},
]


def load_experimental_metrics() -> Optional[Dict[str, Any]]:
    """Load our independently measured experimental metrics from reports/model_metrics.json."""
    metrics_path = REPORTS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_confusion_matrices() -> Optional[Dict[str, Any]]:
    """Load confusion matrix records from reports/confusion_matrices.json."""
    cm_path = REPORTS_DIR / "confusion_matrices.json"
    if not cm_path.exists():
        return None
    try:
        with open(cm_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_feature_importances() -> Optional[Dict[str, Any]]:
    """Load feature importance scores from reports/feature_importance.json."""
    fi_path = REPORTS_DIR / "feature_importance.json"
    if not fi_path.exists():
        return None
    try:
        with open(fi_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_paper_vs_experimental_comparison() -> pd.DataFrame:
    """Build a side-by-side comparative table between paper values and our experimental run."""
    paper_df = pd.DataFrame(PAPER_REPORTED_RESULTS).copy()
    paper_df["source"] = "Paper Reported (Prajapati et al.)"

    exp_payload = load_experimental_metrics()
    if exp_payload and "metrics" in exp_payload:
        exp_df = pd.DataFrame(exp_payload["metrics"]).copy()
        exp_df["source"] = "Our Experimental Run (Synthetic Data)"
        # Normalize model names for easy alignment
        name_map = {
            "HistGradientBoosting (XGBoost Fallback)": "XGBoost",
            "Proposed Hybrid (soft voting)": "Proposed Hybrid AI Model"
        }
        exp_df["aligned_model"] = exp_df["model"].replace(name_map)
        paper_df["aligned_model"] = paper_df["model"]

        combined = []
        for p_row in paper_df.to_dict(orient="records"):
            m_name = p_row["aligned_model"]
            match = exp_df[exp_df["aligned_model"] == m_name]
            if not match.empty:
                e_row = match.iloc[0].to_dict()
                combined.append({
                    "Model": m_name,
                    "Paper Accuracy (%)": p_row["accuracy"],
                    "Our Accuracy (%)": e_row["accuracy"],
                    "Diff Acc (%)": round(e_row["accuracy"] - p_row["accuracy"], 2),
                    "Paper F1 (%)": p_row["f1_score"],
                    "Our F1 (%)": e_row["f1_score"],
                    "Diff F1 (%)": round(e_row["f1_score"] - p_row["f1_score"], 2),
                    "Paper Precision (%)": p_row["precision"],
                    "Our Precision (%)": e_row["precision"],
                    "Paper Recall (%)": p_row["recall"],
                    "Our Recall (%)": e_row["recall"]
                })
        if combined:
            return pd.DataFrame(combined)

    return paper_df


def plot_metrics_comparison_chart(metrics_df: pd.DataFrame, title: str = "Model Performance Comparison") -> go.Figure:
    """Create grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score."""
    fig = go.Figure()

    colors = {
        "accuracy": "#3B82F6",    # Blue
        "precision": "#10B981",   # Emerald
        "recall": "#F59E0B",      # Amber
        "f1_score": "#8B5CF6"     # Purple
    }

    labels = {
        "accuracy": "Accuracy (%)",
        "precision": "Precision (%)",
        "recall": "Recall (%)",
        "f1_score": "F1-Score (%)"
    }

    for metric_col, color in colors.items():
        if metric_col in metrics_df.columns:
            fig.add_trace(go.Bar(
                name=labels[metric_col],
                x=metrics_df["model"],
                y=metrics_df[metric_col],
                marker_color=color,
                text=metrics_df[metric_col].apply(lambda v: f"{v:.1f}%"),
                textposition="auto"
            ))

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Model Architecture",
        yaxis_title="Score Percentage (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    return fig


def plot_confusion_matrix_heatmap(model_name: str, cm_data: Dict[str, Any]) -> go.Figure:
    """Create annotated confusion matrix heatmap."""
    matrix = cm_data.get("matrix", [[cm_data.get("tn", 0), cm_data.get("fp", 0)],
                                    [cm_data.get("fn", 0), cm_data.get("tp", 0)]])

    z = np.array(matrix)
    x_labels = ["Predicted: No Purchase (0)", "Predicted: Purchase (1)"]
    y_labels = ["Actual: No Purchase (0)", "Actual: Purchase (1)"]

    annotations = []
    cell_names = [["True Negative (TN)", "False Positive (FP)"],
                  ["False Negative (FN)", "True Positive (TP)"]]

    for i in range(2):
        for j in range(2):
            val = z[i][j]
            label = cell_names[i][j]
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=f"<b>{val}</b><br><span style='font-size:11px;color:#4B5563'>{label}</span>",
                showarrow=False,
                font=dict(color="#111827", size=14)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        showscale=True,
        reversescale=False
    ))

    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        annotations=annotations,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=380,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    return fig


def plot_feature_importance_chart(model_name: str, feat_dict: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart of feature importances."""
    sorted_feats = sorted(feat_dict.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].replace("_", " ").title() for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color="#2563EB",
        text=[f"{v:.3f}" for v in values],
        textposition="auto"
    ))

    fig.update_layout(
        title=f"Predictive Feature Importance — {model_name}",
        xaxis_title="Relative Importance Weight",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=40, t=50, b=40)
    )
    return fig


def plot_radar_comparison(df: pd.DataFrame, title: str = "Multivariate Model Profile") -> go.Figure:
    """Create radar / spider chart comparing models across all metrics."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig = go.Figure()

    palette = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

    for idx, row in df.iterrows():
        values = [row.get("accuracy", 0), row.get("precision", 0), row.get("recall", 0), row.get("f1_score", 0)]
        values.append(values[0])  # close the radar loop
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["model"],
            line_color=palette[idx % len(palette)],
            opacity=0.6
        ))

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[50, 100])),
        showlegend=True,
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
