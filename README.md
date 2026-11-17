# ReopenRisk — Support Ticket Reopen Prediction & SHAP Explainability

An end-to-end machine learning pipeline and interactive Streamlit app that predicts whether a resolved customer support ticket is likely to reopen—and surfaces the exact feature-level reasons using **SHAP**.

## Problem & Approach

- **Class Imbalance (~13% positive class):** Solved using cost-sensitive learning (`scale_pos_weight`) in XGBoost and Precision-Recall threshold optimization rather than relying on misleading raw accuracy.
- **Explainable Triage:** Uses `shap.TreeExplainer` waterfall plots so support leads can see _why_ a ticket was flagged (e.g., high escalation count, brief closing note, slow first response) before closing it.

## Project Structure

```text
reopen-risk/
├── data/tickets.csv              # Ticket dataset (5,000 records)
├── models/reopen_pipeline.pkl    # Serialized ColumnTransformer + XGBoost + threshold
├── src/generate_data.py          # Synthetic imbalanced support log generator
├── src/train.py                  # Training, threshold tuning, and model serialization
├── app.py                        # Streamlit prediction & SHAP waterfall UI
└── requirements.txt              # Deployment dependencies
```
