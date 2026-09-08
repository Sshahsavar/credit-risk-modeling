# Credit Risk Modeling: PD Scorecard, Model Benchmarking & Portfolio Cut-off

Two notebooks built on the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data) (307,511 loan applications linked to credit bureau history).

| Notebook | What it does |
| --- | --- |
| `End_to_End_Credit_Risk_Modeling.ipynb` | Full PD pipeline: relational aggregation → WOE scorecard → LightGBM challenger → SHAP → cut-off optimization |
| `Benchmarking Classification Algorithms for Credit Risk.ipynb` | Head-to-head comparison of seven classifiers on the same data, scored on AUC and training time |

**Stack:** Python, Pandas, Scikit-Learn, LightGBM, XGBoost, CatBoost, scorecardpy, SHAP, SciPy

---

## Notebook 1: End-to-End PD Modeling

### Step 1 — Relational aggregation

The bureau table holds one row per past loan per client. I one-hot encoded its categorical columns, then grouped by client ID and aggregated numeric columns with mean/max/min/sum and encoded columns with mean (giving the share of each credit type in a client's history). Left-joining this back onto the application table produced 196 columns for the full population.

Clients with no bureau history come through as all-NaN. I kept them. A thin credit file is itself a risk signal, and dropping those rows would bias the training set toward established borrowers.

### Step 2 — Feature engineering and WOE

Three repayment-capacity ratios:

- `CREDIT_TO_INCOME_RATIO` — loan size against earnings
- `ANNUITY_TO_INCOME_RATIO` — share of income going to the payment
- `EMPLOYED_TO_AGE_RATIO` — share of life spent employed

I then selected six features for modeling (the two external bureau scores, two of the ratios, age, and one aggregated bureau feature) and ran WOE binning with `scorecardpy` to rank them by Information Value.

Two results worth noting:

- **Monotonic separation.** `EXT_SOURCE_3` bins run cleanly from 19.5% default probability down to 3.3%.
- **Missingness carries signal.** Roughly 20% of applicants have no `EXT_SOURCE_3`. That bin defaults at 9.3%, above the 8% portfolio baseline. WOE turns the missing group into a measurable risk bucket instead of an imputation problem.

### Step 3 — Imbalance and benchmarking

Defaults are about 8% of the population, an imbalance ratio of 11.39. I used cost-sensitive weights (`scale_pos_weight`, `class_weight='balanced'`) rather than SMOTE, since synthetic minority samples distort the probability calibration that downstream pricing depends on.

| Model | Input | ROC-AUC |
| --- | --- | --- |
| WOE Logistic Regression (baseline) | WOE-transformed | 0.7139 |
| LightGBM (challenger) | Raw features | 0.7246 |

### Step 4 — Risk metrics and explainability

- **KS statistic: 33.60** (`scipy.stats.ks_2samp` on predicted probabilities split by actual label). Above 30 is the usual bar for rank-ordering strength in retail credit.
- **SHAP.** `TreeExplainer` on a 2,000-row test sample. The external bureau scores dominate, as expected. The interesting one is `ANNUITY_TO_INCOME_RATIO`: Information Value of 0.005, effectively useless to a linear model, but SHAP ranks it 5th globally. LightGBM found a non-linear debt-burden effect the scorecard structurally cannot represent. This is the concrete argument for a challenger model, not just a higher AUC.

### Step 5 — Cut-off optimization

AUC and KS don't set policy. I translated probabilities into portfolio economics on a 10,000-applicant sample:

- Loan amount: $15,000
- Interest yield on performing loans: 12%
- Loss Given Default: 60%

Approving a good client earns interest; approving a bad one loses 60% of principal; rejecting earns nothing. Sweeping the probability cut-off traces the growth-versus-loss trade-off directly in dollars.

> **Known limitation.** The current sweep runs from 0.05 to 0.50, and net profit is still increasing at the upper bound — so the reported optimum sits on the edge of the grid, not at a true interior maximum. Under these loan economics the interest earned on marginal approvals keeps outrunning the losses. Extending the grid is the next fix; the more useful takeaway is that the optimal cut-off is highly sensitive to the LGD and interest assumptions, which in practice come from the business, not the modeler.

---

## Notebook 2: Algorithm Benchmark

Seven classifiers on the same six features and the same train/test split, all with cost-sensitive weights.

Each model gets the input its math requires, so the comparison measures the algorithm rather than the preprocessing:

- **Tree ensembles** (Random Forest, LightGBM, XGBoost, CatBoost) — raw, unscaled features, with native missing-value handling
- **Logistic Regression** — WOE-transformed
- **LASSO and SVM** — standardized, so the L1 penalty and the margin geometry apply evenly across features

| Rank | Model | ROC-AUC | Training time |
| --- | --- | --- | --- |
| 1 | CatBoost | 0.7251 | 13.0s |
| 2 | LightGBM | 0.7246 | 5.8s |
| 3 | XGBoost | 0.7234 | — |
| 4 | Random Forest | 0.7182 | 67.7s |
| 5 | Logistic Regression (WOE) | 0.7138 | 1.0s |
| 6 | LASSO | 0.6970 | 0.7s |
| 7 | SVM | 0.6447\* | 483.1s |

\* SVM ran with `max_iter=2000` to keep the benchmark tractable and likely stopped before convergence. Treat this score as a runtime-constrained result, not a fair measure of the algorithm.

**Conclusion.** Boosting captures non-linear structure the linear models can't reach, but the gap between first and fifth place is under 0.012 AUC — feature quality matters more than algorithm choice on this data. LightGBM is the production pick: it matches CatBoost's accuracy in under half the training time.

---

## Repository

```
credit-risk-modeling/
├── End_to_End_Credit_Risk_Modeling.ipynb
├── Benchmarking Classification Algorithms for Credit Risk.ipynb
└── README.md
```

Both notebooks pull the dataset through `kagglehub` and expect `KAGGLE_USERNAME` and `KAGGLE_KEY` to be set (Colab Secrets, or environment variables locally). You'll need to accept the competition rules on Kaggle first.
