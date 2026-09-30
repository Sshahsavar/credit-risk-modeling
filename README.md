# Credit Risk Analysis on Lending Club Loans

This project goes through the main steps of credit risk measurement on one real portfolio: **2.26 million US consumer loans issued by Lending Club from 2007 to 2018**.

It starts with raw data. It then measures loss given default (LGD), builds and validates a probability of default (PD) model, runs a macroeconomic stress test, and ends with an investor's view: buying loans every month and financing them with a securitization.

Each notebook follows the same structure:
* what the part does and how it connects to the other parts;
* why it matters for credit risk;
* the method, the math and the code;
* the result of every step, based on the actual run.

---

## Project Map

The project is built around the expected loss formula:

$$EL = PD \times LGD \times EAD$$

| Part | Notebook | Main question | Main result |
|---|---|---|---|
| 0 | `Part 0 - Credit Risk Data Foundation.ipynb` | What data do we need, and how do we prepare it? | 2,260,668 clean loans with outcome, exit date, exposure and recoveries; unemployment for every state, 2000–2026 |
| 1 | `Part 1 - Loss Given Default (LGD) Modeling.ipynb` | How much do we lose when a borrower defaults? | LGD 91.2%; +0.34 points of LGD per +1 point of unemployment |
| 2 | `Part 2 - Benchmarking Classification Algorithms for Credit Risk.ipynb` | Which model predicts default best? | Gradient boosting leads (AUC ≈ 0.670) ahead of the WOE scorecard (0.653); Lending Club's own sub-grade is still best (0.679) |
| 3 | `Part 3 - End-to-End Credit Risk Modeling (PD & Portfolio Optimization).ipynb` | How do we build, check and use a PD model? | LightGBM AUC 0.672, KS 25.0; stable population (PSI 0.0002) but about 10% more defaults than predicted; break-even PD 26.8% |
| 4 | `Part 4 - Macroeconomic Credit Stress Testing.ipynb` | How much would the portfolio lose in a recession? | Baseline 9-quarter loss 9.5% (\$868M); validation shows the PD model understates the effect of the economy |
| 5 | `Part 5 - Securitization & Forward Flow Analysis.ipynb` | Did purchased loans perform as expected, and who carries the loss? | Actual loss 1.03× expected; senior notes break even at 7.5× expected loss, equity at 1.8× |

**How the parts connect**

```
Part 0  Data foundation ──► clean loan table + macro data
   │
   ▼
Part 1  LGD + downturn LGD ──► lgd_satellite.json (used in Parts 3, 4 and 5)
   │
   ▼
Part 2  Which PD method? ──► champion (scorecard) and challenger (LightGBM)
   │
   ▼
Part 3  PD model: calibration, score, stability, profit cut-off (uses the LGD from Part 1)
   │
   ▼
Part 4  Stress test: PD × LGD × EAD under recession scenarios
   │
   ▼
Part 5  Investor view: forward flow monitoring, securitization, tranche stress
```

**Why LGD comes before PD.** LGD needs only the defaulted loans, and its result is used by three later parts: the profit-based cut-off (Part 3), the stress test (Part 4) and the securitization (Part 5). Measuring it first keeps the run order simple: every part uses only the outputs of the parts before it.

---

## Data

| Source | Content | How to get it |
|---|---|---|
| [All Lending Club loan data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | 2.26M loans, 2007–2018: application data, outcome, payments and recoveries | Free Kaggle account; file `accepted_2007_to_2018Q4.csv` |
| [FRED, Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/) | US and state unemployment rates, US real GDP growth | Downloaded by Part 0, no API key needed |

Why this data works for the whole project:
* **PD:** a default flag and application data.
* **LGD:** the balance at default and the recoveries.
* **EAD:** the loan amount, rate, term and instalment.
* **Stress testing:** issue and payment dates, and the borrower's state, to link each loan to the economy.

The data is not included in this repository.

---

## Methods by Part

**Part 0: Data foundation.**
* **Default definition:** a charged-off loan, dated 4 months after the last payment (about 90 days past due).
* **Exit dates:** every loan is dated as default, prepaid, matured or still open (censored).
* **Leakage control:** columns from after the loan was issued are marked and kept out of PD models.
* **Macro data:** quarterly state unemployment and US GDP growth.

**Part 1: LGD.**
* **Measurement:** workout LGD with discounted recoveries, with a correction for unfinished recoveries.
* **Five models, out of time:** grade average, OLS, fractional logit (hand-coded IRLS with robust standard errors), a two-stage model and LightGBM.
* **Downturn LGD:** a macro model linking LGD to unemployment.

**Part 2: PD algorithm benchmark.**
* **Sample and test:** 36-month loans 2010–2015; training on 2010–2014 and an out-of-time test on 2015.
* **Seven methods:** logistic regression on WOE values, LASSO, SVM, Random Forest, LightGBM, XGBoost and CatBoost.
* **Measures:** AUC, Gini, KS, Brier score and training time. Lending Club's sub-grade is the external benchmark.

**Part 3: PD scorecard and credit decisions.**
* **Variable selection:** WOE and Information Value.
* **Models:** WOE scorecard (champion) vs. LightGBM (challenger), with no class weights.
* **Checks:** calibration and credit-score scaling (points to double the odds), then PSI and CSI stability.
* **Explainability:** SHAP values.
* **Profit-based cut-off:** uses each loan's real cash flows and the LGD from Part 1, checked against the break-even PD $p^* = G/(G+C)$.

**Part 4: Macroeconomic stress testing.**
* **PD model:** a discrete-time hazard model on a loan-quarter panel, with a competing prepayment model.
* **Backtest:** out-of-time, against a model without economic variables.
* **Scenarios:** baseline, adverse and severely adverse, with a 10% unemployment peak as in the Federal Reserve's 2025 scenario, translated to states.
* **Loss projection:** 9 quarters of PD × LGD × EAD.
* **Analysis:** PD vs. LGD effect, sensitivity and a reverse stress test.

**Part 5: Forward flow and securitization.**
* **Forward flow:** monthly loan purchases with eligibility rules and concentration limits.
* **Expected loss at purchase:** PD × EAD × LGD from static-pool curves.
* **Monitoring:** monthly cash flows rebuilt for every loan; actual vs. expected loss by batch.
* **Financing:** a three-tranche waterfall with overcollateralization, a reserve account and triggers, plus a warehouse borrowing base.
* **Stress and break-even:** tranche stress tests and break-even loss multiples.
* **Automation:** an automated monthly investor report.

---

## Key Findings

1. **Unsecured LGD is high and moves with the economy.** About 91% of the balance is lost after default, and LGD is about 3 points higher in high-unemployment periods.
2. **Out-of-time testing matters.** The 2015 loans defaulted more than the 2010–2014 loans (14.9% vs. 13.1%). Every PD model ranked borrowers well but predicted about 10% too few defaults.
3. **The population did not change, the behavior did.** The score's PSI is 0.0002, so the 2015 borrowers looked like earlier borrowers. This is concept drift, and the right response is to recalibrate the model, not to rebuild it.
4. **Risk-based pricing had already done most of the work.** With real interest income and a 91% LGD, a loan is profitable up to a PD of 26.8%. Almost no approved loan is above that level, so a stricter cut-off adds only 0.7% profit.
5. **The stress test is a model validation case study.** The severe scenario raises the loss only from 9.51% to 9.73%. The validation traces this to a missing vintage effect: lending became riskier while unemployment was falling, and the model confuses the two trends. The notebook documents the diagnosis and a plan to fix it.
6. **The investor's view confirms the 2015 weakness.** Forward-flow monitoring flags 6 batches, 5 of them from 2015. The senior notes are very safe (break-even at 7.5× expected loss), but equity carries the risk: it loses money at 1.8× expected loss and is sensitive to prepayment and interest income.

---

## How to Run (Google Colab)

1. Download `accepted_2007_to_2018Q4.csv` from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and upload it to a Google Drive folder.
2. In Colab, open the **Secrets** panel (key icon in the left sidebar), add a secret named `credit_ris_DIR` with the full path of that folder (for example `/content/drive/MyDrive/credit-risk-data`), and allow notebook access. All notebooks read the folder path from this secret, so no personal path is stored in the code.
3. Run the notebooks in number order: **Part 0 → Part 1 → Part 2 → Part 3 → Part 4 → Part 5.** Each part uses only the outputs of the parts before it.

Files each part saves to the Drive folder:

| File | Created by | Used by |
|---|---|---|
| `lc_clean.pkl` | Part 0 | Parts 1–5 |
| `macro_state_q.csv`, `macro_us_q.csv`, `fred/` | Part 0 | Parts 1 and 4 |
| `as_of.txt` | Part 0 | Parts 1, 4 and 5 |
| `lgd_satellite.json` | Part 1 | Parts 3, 4 and 5 |
| `part5_investor_reports_actual.csv` | Part 5 | investor report history |

**Run time on standard Colab:**
* Part 0: about 3 minutes.
* Part 2: about 15 minutes (SVM alone takes about 7).
* Parts 1, 3 and 5: about 5 minutes each.
* Part 4: about 10 minutes.

**Packages:**
* **Already in Colab:** pandas, NumPy, SciPy, scikit-learn, Matplotlib, LightGBM, XGBoost.
* **Installed by the notebooks:** scorecardpy, CatBoost and SHAP.

---

## Limitations

* **Only approved loans.** Rejected applications are not in the data, so model performance on all applicants is unknown (reject inference).
* **Recovery timing is not reported.** LGD discounting uses an assumed 12-month workout (LGD range 89.8–93.4%).
* **Few loans from the 2008–09 recession.** This weakens the downturn LGD and the economic effect in the PD model.
* **Simplified economics.** Funding and operating costs are left out of the profit analysis. The stress test assumes no new loans. The securitization structure is an example, not a real Lending Club deal.

---

Tech stack: Python · pandas · NumPy · SciPy · scikit-learn · LightGBM · XGBoost · CatBoost · scorecardpy · SHAP · Matplotlib · Google Colab · FRED

*Data: Lending Club loan data via Kaggle; macroeconomic data from the Federal Reserve Bank of St. Louis (FRED). This project is for educational and portfolio purposes.*
