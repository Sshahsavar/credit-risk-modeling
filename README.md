# End-to-End Credit Risk Modeling: Probability of Default (PD) & Portfolio Optimization

An end-to-end, production-ready credit risk modeling framework built on relational consumer banking data. This repository benchmarks traditional regulatory-compliant scorecards against modern gradient boosting architectures while optimizing business decision thresholds for portfolio profitability.

---

## Project Overview

In retail banking, credit scoring models must satisfy two competing constraints: predictive power for loss reduction and strict regulatory transparency (e.g., SR 11-7, GDPR). 

This project utilizes the [Home Credit Default Risk Dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data) (300,000+ loan applications) to:
1. Aggregate multi-table relational bureau histories into a unified analytical view.
2. Compare a traditional **WOE Logistic Regression Scorecard** against a **LightGBM Challenger Model**.
3. Evaluate rank-ordering performance using the **Kolmogorov-Smirnov (KS) Statistic**.
4. Interpret non-linear feature dynamics using **SHAP values**.
5. Simulate expected credit losses and optimize loan decision cut-off thresholds for maximum net revenue.

---

## Technical Architecture & Methodology
