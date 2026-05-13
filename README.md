# EasyVisa Visa Certification Classification

**Domain:** Immigration · Policy Analytics  
**Tools:** Python, scikit-learn, XGBoost  

---

## The Problem

The Office of Foreign Labor Certification processes hundreds of thousands of visa applications annually. The goal: build a model that predicts whether a visa application will be certified or denied, and identify the factors that drive that outcome.

Both types of errors carry real costs. A wrongly certified application displaces a qualified US worker. A wrongly denied application loses a qualified foreign worker. F1 score — balancing precision and recall — is the right metric here.

---

## The Approach

Full ensemble comparison across six model families, all tuned with GridSearchCV:

- Decision Tree (baseline, pre and post pruning)
- Bagging Classifier
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

A Stacking Classifier combining the best performers was also evaluated as a meta-learner.

Balanced class weights applied throughout to ensure the model treats certification and denial equally during training.

---

## Results

| Model | F1 (Test) |
|---|---|
| XGBoost tuned (final) | **0.821** |
| Gradient Boosting (default) | 0.821 |
| AdaBoost tuned | 0.819 |
| Tuned Bagging | 0.812 |
| Random Forest tuned | 0.811 |

XGBoost and default Gradient Boosting were effectively tied. XGBoost was selected for its slightly better train/test consistency.

---

## Key Finding

Education level is by far the strongest predictor — high school education alone accounts for 35-40% of relative feature importance. Job experience is second. Together they suggest the model is largely capturing qualification signals that align with what the certification process is designed to assess.

---

## What This Demonstrates

- Systematic comparison across ensemble model families
- Metric selection tied to business cost structure
- Feature importance interpretation connected to domain context
- Stacking as a meta-learning strategy
