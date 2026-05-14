# EasyVisa Visa Certification Classification

**Domain:** Immigration · Policy Analytics  
**Tools:** Python, scikit-learn, XGBoost  

---

## The Problem

The Office of Foreign Labor Certification processes hundreds of thousands of visa applications annually. The stated purpose is to certify foreign workers only where US workers are not available at prevailing wages — protecting both domestic labor and foreign applicants from being underpaid.

The task here was to build a model that predicts whether an application gets certified or denied, and identify what drives that outcome. Both error types carry real costs. A wrongly certified application potentially displaces a qualified US worker. A wrongly denied application loses a qualified foreign worker. F1 score, balancing precision and recall, is the right metric.

---

## What the Data Shows

A few patterns in the data are worth noting before getting to the model.

Europeans are certified at nearly 80%. Applicants from Asia — who make up by far the largest share of applications — are certified at 65%. South Americans have the lowest approval rate at 58%.

Higher education correlates strongly with approval: 87% of doctorate holders are certified, 79% of master's degree holders, and 66% of applicants with only a high school education are denied. That pattern aligns with the intent of the program.

What doesn't align as cleanly is the prevailing wage data. Doctorate holders consistently show lower prevailing wages than applicants with lower education levels across multiple wage units and regions. If the wage floor is designed to protect workers, the data doesn't obviously show it working that way.

---

## The Approach

Full ensemble comparison across six model families, all tuned with GridSearchCV:

- Decision Tree (baseline, pre and post pruning)
- Bagging Classifier
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

A Stacking Classifier combining the best performers was also evaluated as a meta-learner. Balanced class weights applied throughout to ensure the model treats certification and denial equally during training.

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

## What the Model Found

Education level dominates — high school education alone accounts for 35-40% of relative feature importance. Job experience is second. The model is largely capturing the qualification signals the certification process is designed to assess.

Some findings are harder to interpret cleanly. Doctorate holders are approved at 87% but consistently show lower prevailing wages than applicants with lower education levels across multiple wage units and regions. A doctorate holder in the data earns less than a high school graduate in several cases. Whether that reflects the job categories being filled, the industries involved, or something else in the data isn't clear from the features available.

---

## What This Demonstrates

The model performs well technically. The more interesting exercise was sitting with what the data raises but can't answer.

Doctorate holders are approved at 87% but earn less than applicants with lower education in several cases. Does a doctorate from one country carry the same weight as one from another? The data treats all advanced degrees equally, but should it?

Having only a high school education is the single strongest predictor of denial, yet a meaningful number of applicants with only a high school diploma and no job experience still get certified. Under what circumstances does that make sense?

For US workers who took on significant student debt to achieve the credentials the program rewards, seeing those same credentials approved at lower wage levels for foreign applicants raises questions about whether the prevailing wage protection is doing what it's supposed to do.

The stated purpose of the program is to certify foreign workers only where equivalent US workers aren't available, at wages that don't undercut the market. Beyond education level, the data doesn't clearly show that alignment. That may reflect limitations in the available features, or it may reflect something worth examining in the program itself.
