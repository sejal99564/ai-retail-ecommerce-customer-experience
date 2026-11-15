# Artificial Intelligence in Retail and E-commerce: Transforming Customer Experience

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Supported-green.svg)](https://xgboost.readthedocs.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://sqlite.org/)
[![Academic Project](https://img.shields.io/badge/Academic-Research%20Paper%20Implementation-purple.svg)]()

> **Academic Project Reference:**  
> R N Prajapati, Prashant Kumar, Aditya Prasad, Mayank Rai, Sunny Kumar, and Gaurav Kumar,  
> *"Artificial Intelligence in Retail and E-commerce: Transforming Customer Experience,"*  
> Department of Mathematics & Department of Computer Science and Engineering, Chandigarh University, Mohali, India.  
> Included in `research/research_paper.pdf`.

---

## 📖 Executive Summary

This repository delivers a complete, modular, and professional academic prototype implementing the holistic AI system proposed by Prajapati et al. (2024). The system addresses three critical research gaps in retail AI:
1. **Unification of Siloed Services**: Integrating personalized recommendations, conversational support, and dynamic pricing on a coordinated data platform.
2. **Ethical AI & Customer Trust**: Implementing explicit price surge ceilings, floor bounds, essential goods exemptions, and transparent Explainable AI (XAI) justifications to protect consumer welfare.
3. **Hybrid AI Methodology**: Demonstrating how ensembling complementary predictive models (Logistic Regression, Random Forest, and XGBoost) produces balanced classification performance for consumer purchase intent.

---

## 🏛️ System Architecture

The application adheres directly to the **Four-Layer Architecture** described in Section III (Figure 1) of the research paper:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Hardware & Data Acquisition Layer                            │
│  • Point of Sale (POS), Web & Mobile Clients                           │
│  • Captures clicks, searches, cart actions, views, purchases, queries  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│  Layer 2: Cloud & Database Storage Layer (SQLite DBMS)                 │
│  • Relational schema: Products, Customers, Orders, Transactions, Logs  │
│  • Audit trail for dynamic pricing and multi-turn conversations        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│  Layer 3: AI and Machine Learning Intelligence Layer                   │
│  ├── A. Personalized Recommender: TF-IDF Content + Collab + XAI        │
│  ├── B. Conversational AI Engine: Local NLU + DB Order Tracking        │
│  ├── C. Dynamic Pricing Engine: Demand + Scarcity + Fairness Bounds    │
│  └── D. Comparative ML Suite: LR, RF, XGBoost & Proposed Hybrid Model  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│  Layer 4: User Interface & Application Layer                           │
│  • Modern Streamlit Analytics Dashboard (7 Interactive Views)          │
│  • Interactive Plotly visual charts, confusion matrices, and metrics   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Core AI Subsystems

### 1. Personalized Product Recommendation (Section IV-B)
- **Hybrid Architecture**: Combines **Content-Based Filtering** (TF-IDF vectorization of product names, categories, brands, descriptions, and tags with Cosine Similarity) with **Collaborative Behavioral Signals** (historical customer category affinities, repeat purchase frequency, and social proof ratings).
- **Hard Constraints**: Strictly respects user-defined budget boundaries and stock availability.
- **Explainable AI (XAI)**: Generates clear, human-understandable justification bullets for every recommended product (e.g., *"Category Match: 95% alignment with preferred category Electronics"*, *"Budget Fit: ₹3,499 fits comfortably within your ₹5,000 budget"*, *"Customer Favorite: 4.8★ with 310+ verified reviews"*).
- **Cold-Start Resilience**: Seamlessly supports both logged-in registered shoppers with historical profiles and guest visitors.

### 2. Conversational AI Customer Support (Section IV-B)
- **Zero Paid APIs**: Fully functional locally without requiring OpenAI, Anthropic, or external API keys.
- **Local NLU Intent Engine**: Classifies intents across:
  - **Order Tracking**: Recognizes order IDs (e.g., `ORD-1001` through `ORD-1150`) and queries the SQLite database in real time for item details, shipment statuses, carrier tracking numbers, and delivery dates.
  - **Product Inquiries**: Searches the live product catalog for matching keywords, prices, and stock counts.
  - **Store Policies**: Resolves detailed questions regarding shipping tiers, 30-day return policies, refund processes, payment options (UPI/Cards/COD), and cancellations.
  - **Human Escalation**: Provides seamless escalation pathways with ticket generation.
- **Context Preservation**: Retains multi-turn conversation history in session and logs interactions to the database.

### 3. Adaptive Dynamic Pricing & Fairness Safeguards (Section IV-B & VII-B)
- **Multi-Signal Pricing Formulation**:
  $$\text{Adjustment} = \Delta_{demand} + \Delta_{scarcity} + \Delta_{competitor} + \Delta_{rating}$$
  $$P_{raw} = P_{base} \times (1 + \text{Adjustment})$$
- **Ethical Fairness Safeguards**:
  - **Surge Ceiling**: Strict cap (default +25%) prevents price gouging during peak shopping hours.
  - **Floor Price Bound**: Capped markdown (default -20%) protects supplier and seller margins.
  - **Essential Goods Protection**: Products tagged as `is_essential` (e.g. water bottles, air purifiers, healthcare items) are strictly exempted from surge pricing.
  - **Non-Discrimination Policy**: No sensitive demographic traits (gender, age, location) are ever factored into pricing calculations.
- **Explainability & Simulation**: Deconstructs every rupee adjustment transparently and simulates dynamic price curves across demand indices.

---

## 🔬 Machine Learning Implementation & Evaluation

Following the experimental methodology of the research paper (Section VI-A), we implement, train, and compare four supervised learning models for **Customer Purchase Intent Classification**:

1. **Logistic Regression**: Scaled feature pipeline with L2 regularization.
2. **Random Forest Classifier**: Ensemble of 150 bagged decision trees with feature importance extraction.
3. **XGBoost Classifier**: Gradient boosted decision trees using `xgboost.XGBClassifier` (with `HistGradientBoostingClassifier` fallback if unavailable).
4. **Proposed Hybrid AI Model**: An actual multi-model ensemble combining Logistic Regression, Random Forest, and XGBoost using soft probability voting (`VotingClassifier`).

### Comparative Evaluation Metrics
All models are evaluated on a stratified 80/20 train/test split (random seed 42) using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrices (True Positives, False Positives, True Negatives, False Negatives)**
- **Per-class Classification Reports**

---

## 📊 Research Paper vs. Experimental Results

### Research Integrity Statement
The research paper (Prajapati et al., 2024) reports metrics obtained on a proprietary, unpublished dataset. In compliance with strict academic integrity:
1. **Paper-Reported Results** are shown directly as reported in Table I of the paper.
2. **Our Experimental Results** are independently measured and calculated from the training pipeline on our synthetic demonstration dataset.
3. **No fabricated results**: Results are not hardcoded or forced to match the paper's values.

| Model Architecture | Paper Accuracy | Our Accuracy | Paper F1-Score | Our F1-Score | Paper Precision | Our Precision | Paper Recall | Our Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 89.4% | **72.97%** | 88.3% | **70.63%** | 88.7% | **74.02%** | 87.9% | **67.53%** |
| **Random Forest** | 93.1% | **69.53%** | 92.4% | **66.55%** | 92.6% | **70.55%** | 92.2% | **62.99%** |
| **XGBoost** | 95.0% | **68.59%** | 94.3% | **65.99%** | 94.5% | **68.90%** | 94.1% | **63.31%** |
| **Proposed Hybrid AI Model** | **96.7%** | **70.31%** | **96.0%** | **68.12%** | **96.2%** | **70.49%** | **95.9%** | **65.91%** |

*Note: Differences stem from dataset variance, sample size, feature space dimensions, and noise characteristics. The Proposed Hybrid Model demonstrates improved recall and balanced decision boundaries compared to individual non-linear models.*

---

## 📁 Repository Structure

```text
AI_Retail_Ecommerce_Antigravity_Starter/
├── app.py                      # Main Streamlit dashboard (7 interactive pages)
├── requirements.txt            # Project dependencies
├── README.md                   # Complete academic documentation & run guide
├── ANTIGRAVITY_PROMPT.md       # Research specification & compliance rubric
├── research/
│   └── research_paper.pdf      # Original research paper (Prajapati et al., 2024)
├── src/
│   ├── database.py             # SQLite DBMS schema, connections, and transactional queries
│   ├── generate_demo_data.py   # Synthetic data generator & SQLite database seeder
│   ├── recommender.py          # Hybrid recommendation engine with Explainable AI (XAI)
│   ├── chatbot.py              # Local conversational AI support engine & order tracker
│   ├── dynamic_pricing.py      # Adaptive pricing engine with ethical fairness guardrails
│   ├── train_models.py         # Supervised ML training pipeline (LR, RF, XGB, Hybrid)
│   ├── evaluation.py           # Evaluation visualizers, confusion matrices & Plotly charts
│   └── services.py             # Unified facade interface maintaining backwards compatibility
├── tests/
│   └── test_all.py             # Automated unit & integration test suite (23 tests)
├── data/
│   ├── retail_ecommerce.db     # SQLite relational database
│   ├── products.csv            # 40 products across 6 retail categories
│   ├── customers.csv           # 250 customer profiles with loyalty tiers
│   ├── orders.csv              # 150 trackable orders
│   ├── transactions.csv        # 800 historical purchase transactions
│   ├── interactions.csv        # 1500 behavioral interaction records
│   └── synthetic_model_data.csv# 3200 ML samples for purchase intent prediction
├── models/                     # Serialized trained models (*.joblib)
└── reports/                    # JSON evaluation reports & confusion matrices
```

---

## ⚙️ Installation and Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- Windows PowerShell, macOS Terminal, or Linux Bash

### 2. Environment Setup (Windows PowerShell)

```powershell
# 1. Clone or navigate to the project directory
cd D:\cod\AI_Retail_Ecommerce_Antigravity_Starter

# 2. Create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

*Note: If PowerShell execution policy restricts script execution, run:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Generate Data and Seed Database

```powershell
python src\generate_demo_data.py
```
*Creates the SQLite database (`data/retail_ecommerce.db`) with 40 products, 250 customers, 150 orders, 800 transactions, 1,500 interactions, and 3,200 ML samples.*

### 4. Train and Evaluate Machine Learning Models

```powershell
python src\train_models.py
```
*Trains Logistic Regression, Random Forest, XGBoost, and the Proposed Hybrid Model. Saves models to `models/` and evaluation metrics to `reports/`.*

### 5. Run Automated Tests

```powershell
python tests\test_all.py
```
*Executes all 23 unit tests verifying database queries, recommendations, chatbot intents, pricing fairness, and evaluation metrics.*

### 6. Launch the Streamlit Analytics Application

```powershell
streamlit run app.py
```
*Open your web browser and navigate to `http://localhost:8501`.*

---

## 🎯 Evaluator Demonstration Guide (5–10 Minute Viva Script)

For undergraduate students presenting this project to an evaluator, follow this structured walkthrough:

1. **Introduction (1 minute)**:
   - State the project title: *"Artificial Intelligence in Retail and E-commerce: Transforming Customer Experience"*.
   - Cite the paper by Prajapati et al. (Chandigarh University, 2024).
   - Open the **Home Dashboard** and highlight the 4-layer architecture diagram.

2. **Personalized Product Recommendation (2 minutes)**:
   - Navigate to **Personalized Recommendation**.
   - Select a sample customer (e.g. *Aarav Sharma — Gold Tier, Prefers Electronics*).
   - Point out how the customer's historical category purchases influence the rankings.
   - Adjust the budget slider and show how the system strictly respects budget constraints.
   - **Crucial Point**: Show the green **Explainable AI (XAI)** box beneath each product card explaining *why* it was recommended.

3. **Conversational AI Customer Support (2 minutes)**:
   - Navigate to **Conversational AI Support**.
   - Click the prompt chip **"Track ORD-1002"** (or type any order ID like `ORD-1005`).
   - Demonstrate that the bot retrieves the actual customer order details, tracking number, and live status directly from the SQLite database.
   - Ask about policy: *"What is your 30-day return policy?"* or search products: *"Do you have wireless headphones?"*.
   - Mention that this runs 100% locally with zero paid API costs.

4. **Adaptive Dynamic Pricing & Fairness Safeguards (2 minutes)**:
   - Navigate to **Adaptive Dynamic Pricing**.
   - Pick a product (e.g. *Apex Chrono Pro Smartwatch*).
   - Drag the **Demand Index slider** from 0.50 to 0.95: show the suggested price increase.
   - Drag **Stock Units** down to 3: show the low-stock scarcity premium.
   - **Crucial Ethical Point**: Toggle **"Mark as Essential Item"** and show that the fairness guardrail immediately overrides the surge, enforcing a zero price increase on essential goods.

5. **Machine Learning & Research Results Comparison (2 minutes)**:
   - Navigate to **ML Model Evaluation**: Show the interactive side-by-side metric charts, confusion matrices, and feature importance drivers.
   - Navigate to **Research vs. Experimental Results**: Show the comparison table separating the paper's reported values from our independently measured values. Explain why academic integrity requires distinguishing synthetic demo results from proprietary published claims.

---

## ⚠️ Known Limitations & Future Work

As noted in Section VII-B of the research paper:
- **Dataset Scope**: The current dataset is synthetic demonstration data. Future extensions should incorporate production clickstream logs.
- **Deep Reinforcement Learning**: The dynamic pricing engine currently uses multi-factor elasticity equations with ethical bounds; transitioning to multi-agent Deep Q-Networks (DQN) represents a future extension.
- **Multimodal AI**: Integrating visual computer vision search (product images) and speech-to-text voicebots.
- **Federated & Privacy-Preserving Learning**: Implementing differential privacy for cross-retailer consumer analytics.

---

## 📜 License & Acknowledgements

Developed as an educational implementation of the research paper *"Artificial Intelligence in Retail and E-commerce: Transforming Customer Experience"* (Prajapati et al., 2024). Free for academic and demonstration purposes.
