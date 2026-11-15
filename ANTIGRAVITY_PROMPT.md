# Prompt for Google Antigravity

You are working inside the provided repository for the academic project
**“Artificial Intelligence in Retail and E-commerce: Transforming Customer
Experience.”** First read `README.md`, inspect the full repository, and read
`research/research_paper.pdf` if PDF inspection is available. Treat the paper
as the primary source for the project's research scope.

## Main objective
Turn this starter into a polished, fully runnable academic prototype that
demonstrates the paper's integrated framework:
1. Personalized product recommendation
2. Conversational customer support
3. Dynamic pricing
4. Comparative ML evaluation using accuracy, precision, recall, and F1-score

Preserve the paper's terminology and do not claim that features/results exist
unless you implement and test them.

## Critical research-integrity requirements
- Do not hardcode or fabricate model scores to match the paper.
- The paper reports Logistic Regression 89.4% accuracy, Random Forest 93.1%,
  XGBoost 95.0%, and Proposed Hybrid AI Model 96.7%; show these only in a
  clearly labelled “Paper-reported results” comparison.
- Show independently measured application results in a separate
  “Our experimental results” section.
- The paper does not sufficiently specify dataset provenance/size, exact
  target construction, hybrid composition, or all hyperparameters. Do not
  claim exact reproduction.
- Clearly label the included data as synthetic demonstration data until
  replaced with a documented, licensed real dataset.
- If XGBoost is unavailable, either install it safely and update requirements,
  or clearly label the fallback model. Never label a different algorithm as
  XGBoost.
- Do not describe the current FAQ chatbot as an LLM or the pricing heuristic
  as reinforcement learning. Implement those advanced approaches only if
  they can be tested and documented honestly.

## Work to perform
1. Inspect the code and run the application end-to-end.
2. Fix errors, missing imports, path issues, and dependency problems.
3. Improve the UI into a clean, professional retail analytics dashboard.
4. Make recommendation logic more meaningful:
   - accept a customer profile or preference inputs;
   - combine category match, budget fit, rating, and available stock;
   - explain why each product was recommended;
   - handle empty results gracefully.
5. Improve customer support:
   - preserve conversation history;
   - support common intents (shipping, returns, payment, order status,
     product information);
   - state clearly that this is a prototype and does not access live orders.
   - If using an LLM, make API keys optional via environment variables and
     never commit secrets.
6. Improve dynamic pricing:
   - show the input signals and calculation transparently;
   - include a price-change explanation;
   - add guardrails (minimum/maximum price bounds and no sensitive personal
     attributes);
   - label it as a simulation unless a validated model is implemented.
7. Improve the model pipeline:
   - use a reproducible train/test split and fixed random seed;
   - add a confusion matrix and classification report;
   - compare Logistic Regression, Random Forest, XGBoost (or accurately named
     fallback), and a documented hybrid ensemble;
   - save metrics and trained models;
   - ensure the hybrid ensemble is a real implemented model, not a renamed
     single classifier.
8. Add a “Research paper vs implementation” page explaining which paper
   elements are implemented, approximated, or not implemented.
9. Add tests for core recommendation, chatbot, pricing, and model evaluation
   functions. Include exact commands to run tests.
10. Update README with installation, run instructions, architecture,
    dataset schema, limitations, screenshots guidance, and evaluator demo flow.

## Technical constraints
- Prefer Python, Streamlit, Pandas, NumPy, scikit-learn, Plotly, and SQLite
  unless a strong reason exists to change the stack.
- Keep code modular and readable.
- Use type hints and docstrings for important functions.
- Handle missing files and invalid user inputs gracefully.
- Do not require paid services or API keys for the default demo.
- Do not add unrelated features or change the project's research topic.

## Validation before finishing
Run the data generator, training pipeline, automated tests, and Streamlit
startup check. Fix all errors you can reproduce. Provide a concise completion
summary listing:
- files changed;
- features implemented;
- commands executed and their outcomes;
- actual measured metrics and which dataset produced them;
- known limitations and next steps.

Do not say the project is complete if it has not been run and tested.
