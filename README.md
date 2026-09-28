# Credit Risk Modeling: PD, LGD & Macroeconomic Stress Testing

An end-to-end credit risk project on **2.26 million Lending Club consumer loans (2007–2018)**. It covers all three risk parameters of the expected-loss equation, **EL = PD × LGD × EAD**, and links them to the economy through a bottom-up stress test.

Each notebook is written as a tutorial: it states the business question, explains the methodology and the math, runs the code, and closes with conclusions drawn from the actual results.

---

## Project Structure

| Part | Notebook | Question | Headline Result |
|---|---|---|---|
| 0 | `Part 0 - Credit Risk Data Foundation.ipynb` | Build one clean dataset that supports PD, LGD, EAD and stress testing | 2,260,668 loans with outcomes, exit dates, exposures and recoveries; state unemployment for all 51 states, 2000–2026 |
| 1 | `Part 1 - Benchmarking Classification Algorithms for Credit Risk.ipynb` | Which algorithm predicts default best, out of time? | XGBoost / LightGBM / CatBoost lead (AUC ≈ 0.670), ahead of the WOE scorecard (0.653). Lending Club's own sub-grade remains the best benchmark (0.679) |
| 2 | `Part 2 - End-to-End Credit Risk Modeling (PD & Portfolio Optimization).ipynb` | Build a PD scorecard, check calibration, and find the profit-optimal cut-off | LightGBM: AUC 0.672, KS 25.0. Break-even PD of 26.8%. Re-screening already-approved loans adds only +0.7% profit |
| 3 | `Part 3 - Loss Given Default (LGD) Modeling.ipynb` | What share of a defaulted loan is lost, and how does it change in a recession? | LGD = 91.2%, with 21.6% of defaults recovering nothing. +0.34 pp of LGD per +1 pp of unemployment |
| 4 | `Part 4 - Macroeconomic Credit Stress Testing.ipynb` | How much would the loan book lose over 9 quarters in a recession? | Baseline 9-quarter loss of 9.5% ($868M). Validation shows the PD model understates macro sensitivity (see findings) |

**Run order:** Part 0 → Part 3 → Parts 1, 2 and 4. Parts 1 and 2 need Part 0's output; Part 2 also uses the LGD from Part 3. Part 4 needs Parts 0 and 3.

---

## Data

| Source | Content | Access |
|---|---|---|
| [Lending Club loan data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | 2.26M accepted loans, 2007–2018: application data, repayment outcome, recoveries | Free Kaggle account |
| [FRED (St. Louis Fed)](https://fred.stlouisfed.org/) | US and state unemployment rates (`UNRATE`, `{ST}UR`), real GDP growth (`A191RL1Q225SBEA`) | Downloaded automatically, no API key |

**Why Lending Club and not Home Credit?** An earlier version of Parts 1 and 2 used the Home Credit dataset. It works for PD modeling, but it cannot support the rest of the project:
- it has **no recovery data**, so LGD can only be assumed;
- it has **no calendar dates or geography**, so defaults cannot be linked to the economy for stress testing.

Lending Club provides:
- **recoveries** (`recoveries`, `collection_recovery_fee`);
- **exposure at default** (`funded_amnt − total_rec_prncp`);
- **monthly dates** (`issue_d`, `last_pymnt_d`);
- **borrower location** (`addr_state`).

The data is not included in this repository. See **How to Run**.

---

## Methods

**Part 0: Data Foundation**
- Outcome mapping: loan statuses → default / paid / current / delinquent.
- Event timing: default dated at last payment + 4 months (≈ 90 days past due), prepayment vs. maturity, censoring at the March 2019 snapshot.
- An explicit list of post-origination fields that must never enter a PD model (leakage control).
- Quarterly state unemployment, its 4-quarter change, and US GDP growth from FRED.

**Part 1: Algorithm Benchmark**
- Seven algorithms: Logistic Regression (WOE), LASSO, SVM, Random Forest, LightGBM, XGBoost, CatBoost.
- Target: charge-off during the life of 36-month loans issued 2010–2015, all of which had matured by the snapshot.
- **Out-of-time validation:** trained on 2010–2014 (329,719 loans), tested on 2015 (283,026 loans).
- Data routed by algorithm type: raw features for trees, WOE for logistic regression, standardized features for LASSO and SVM.
- Metrics: AUC, Gini, KS, Brier score, training time. Lending Club's own sub-grade serves as an external benchmark.

**Part 2: PD Scorecard & Portfolio Optimization**
- WOE / IV feature selection (bins fitted on training data only).
- WOE logistic scorecard (champion) vs. LightGBM (challenger), with no class weights so the PDs stay calibrated.
- Calibration-in-the-large, calibration by decile, and PDO score scaling (600 points = 50:1 odds, 20 points to double the odds).
- SHAP explainability.
- Profit optimization on **realized cash flows**:
  - each loan's actual interest income until payoff or default;
  - exposure at default × the LGD from Part 3;
  - the optimum checked against the analytical break-even PD, p* = G / (G + C).

**Part 3: LGD**
- Workout LGD with discounted net recoveries.
- Resolution-bias correction (only completed workouts are used).
- Default-weighted vs. exposure-weighted averages.
- Five models compared out of time: grade average, OLS, fractional logit (Papke–Wooldridge, estimated with hand-coded IRLS and sandwich standard errors), a two-stage hurdle model, and LightGBM + SHAP.
- Downturn LGD, both observed and through a macro satellite model.

**Part 4: Stress Testing**
- A discrete-time hazard model on a loan-quarter panel (849,045 rows), with seasoning, borrower risk and lagged state macro variables, plus a competing prepayment hazard.
- An out-of-time backtest against a no-macro benchmark.
- Baseline / adverse / severely adverse scenarios over 9 quarters. The severely adverse peak of 10% unemployment matches the Fed's 2025 scenario.
- National scenarios downscaled to states with state betas.
- A 9-quarter expected-loss projection with amortizing EAD and scenario-dependent LGD.
- A PD-vs-LGD decomposition, a sensitivity curve and a reverse stress test.

---

## Key Findings

1. **Out-of-time testing matters.** The 2015 vintage defaulted more often than 2010–2014 (14.9% vs. 13.1%). Every PD model ranked borrowers well but **under-predicted defaults by about 10%**, a calibration drift that a random split would have hidden.

2. **Gradient boosting beats the scorecard, but not the lender.** LightGBM improves on the WOE scorecard by about 0.02 AUC. Lending Club's own sub-grade, built with richer credit-bureau data, still ranks best.

3. **Risk-based pricing had already done most of the work.** With real interest income and a 91% LGD, a loan is profitable in expectation up to a PD of 26.8%. Almost no approved Lending Club loan exceeds this, so re-screening adds just $1.5M (+0.7%) on a $228M portfolio profit. A PD model earns its value in pricing and in screening new applicants.

4. **Unsecured LGD is high and cyclical.** 91.2% of the defaulted balance is lost, far above the 60% assumed in the original Home Credit version. Loan-level models explain only about 1% of the variation. LGD rises by about 3 pp in high-unemployment periods.

5. **The stress test is a model-validation case study.** The severely adverse scenario raises the 9-quarter loss only from 9.51% to 9.73%. The validation traces this to three findings:
   - a **wrong-signed unemployment coefficient**;
   - **no backtest gain** over a model without macro variables;
   - a **16% under-prediction** of defaults.

   The root cause is **omitted-variable bias**: Lending Club loosened its underwriting while unemployment fell (2010–2018), and the model confuses the two trends. The notebook documents the diagnosis and a remediation plan (vintage effects, state fixed effects, an interim overlay), as a validator would.

---

## How to Run (Google Colab)

1. Download `accepted_2007_to_2018Q4.csv.gz` from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club), unzip it, and upload `accepted_2007_to_2018Q4.csv` to a Google Drive folder.
2. Open each notebook in Colab and set `PROJECT_DIR` in Step 1 to that folder:
   ```python
   PROJECT_DIR = '/content/drive/MyDrive/<your-folder>'
   ```
3. Run the notebooks in order: **Part 0 → Part 3 → Parts 1, 2, 4**. Each notebook saves its outputs to the same Drive folder:

| File | Created by | Used by |
|---|---|---|
| `lc_clean.pkl` | Part 0 | Parts 1–4 |
| `macro_state_q.csv`, `macro_us_q.csv`, `as_of.txt`, `fred/` | Part 0 | Parts 3–4 |
| `lgd_satellite.json` | Part 3 | Parts 2 and 4 |

**Runtime (standard Colab):** Part 0 takes about 3 minutes and Part 3 about 5. Part 1 takes about 15 minutes, of which SVM alone is about 9. Part 2 takes about 5 minutes and Part 4 about 10.

**Packages:** pandas, numpy, scipy, scikit-learn, matplotlib, lightgbm, xgboost (all pre-installed in Colab); scorecardpy, catboost and shap are installed automatically by the notebooks.

---

## Limitations

- **Reject inference.** Only approved loans are observed, so model performance on the full applicant population is unknown.
- **Recovery timing is not reported.** LGD discounting relies on an assumed workout length (sensitivity range 89.8–93.4%).
- **One recession, few loans in it.** Lending Club was very small in 2008–09, which weakens the downturn LGD and the macro sensitivity of the PD model.
- **Simplified economics.** Funding and operating costs are excluded from the profit analysis, and the stress test assumes a static balance sheet with no management actions.

---

## Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · LightGBM · XGBoost · CatBoost · scorecardpy · SHAP · Matplotlib · Google Colab · FRED

---

*Data: Lending Club loan data via Kaggle; macroeconomic data from the Federal Reserve Bank of St. Louis (FRED). This project is for educational and portfolio purposes.*
