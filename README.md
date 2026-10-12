# IAM ContextBench: Implementation and Experiments

**"Evaluating Behavioral Context for Interpretable IAM Policy Risk Scoring in Cloud Environments"**

This repository contains the complete empirical machine learning implementation, three core experiments, statistical evaluations, interpretability decompositions, publication figures, comparison tables, out-of-fold predictions, error analysis, and research documentation for the paper.

Paper link: https://arxiv.org/abs/2610.07345

Peer-reviewed and accepted at IEEE Conference ID# 71863; to appear in the IEEE Xplore proceedings.

---

## 1. Executive Summary & Research Question

**Central Question:**
> *"Does AWS CloudTrail-derived behavioral context provide measurable incremental value for interpretable IAM policy risk prioritization for security analysts beyond policy-centric information?"*

Specifically, can the exact same IAM policy and authorization receive differential analyst-review priority scores when the surrounding real CloudTrail behavior indicates anomalous, unauthorized, or multi-step adversarial activity?

### Conceptual Distinction
- **Intrinsic Security Risk (`gt_security_risk_score`)**: How dangerous is the policy/authorization itself? Invariant across same-policy/same-environment pairs.
- **Analyst Review Priority (`gt_analyst_priority_score`)**: How urgently should a security analyst review this policy event given who acted, what they were doing, and the surrounding AWS behavioral context? **(Primary ML Target)**.

---

## 2. Exactly Three Core Experiments

1. **Experiment 1: Incremental Context Ablation**
   - Evaluates three feature tiers with identical model architecture, hyperparameters, and 5-fold grouped splits:
     - **$M_0$ (Policy-Centric)**: 15 syntax, wildcard, and effective authorization features.
     - **$M_1$ (Policy + Environment)**: 28 features (M0 + AWS environment topology, boundary, privesc paths).
     - **$M_2$ (Full Context)**: 40 features (M1 + CloudTrail event counts, errors, user-agent novelty, role chains).
   - Metrics: MAE, RMSE, Spearman $\rho$, Kendall $\tau$, NDCG@10/25/50, and 95% cluster-bootstrap confidence intervals.

2. **Experiment 2: Same-Policy / Different-Context Paired Test**
   - Conducted strictly on the 208 validated Type-A context pairs (416 samples) with verified identical structural policy hash, environment, and effective authorizations.
   - Tests whether $M_0$ remains invariant (gap $\approx 0$) while $M_2$ sensitively elevates suspicious contexts over benign peers ($M_2\text{-gap} > 0$).
   - Evaluates direction accuracy, ranking accuracy, gap correlation, Wilcoxon signed-rank tests, paired permutation tests (5,000 iterations), and pair-resampled bootstrap CIs.

3. **Experiment 3: Analyst Triage Utility + Interpretability**
   - Simulates constrained analyst review queues ranking out-of-fold alerts.
   - Evaluates Top-$K$ retrieval performance ($K \in \{10, 25, 50\}$) via NDCG@K, Precision@K, Recall@K, and Triage Lift.
   - Computes exact local additive EBM term decompositions:
     $$\text{Score} = \beta_0 + \sum \text{Policy Terms} + \sum \text{Env Terms} + \sum \text{CloudTrail Terms}$$
   - Conducts systematic diagnostic error analysis on 10 representative cases.

---

## 3. How to Reproduce All Results

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Dependencies: `pip install -r requirements.txt`

### Execution
Run the entire research pipeline with:
```bash
python run_all.py
```
or from the repository parent:
```bash
python -m aws_iam_contextbench_ml.run_all
```

To run the automated unit tests:
```bash
python -m unittest discover -s tests -v
```

---

## 4. Key Research Findings Summary

| Evaluation Dimension | Metric | $M_0$ (Policy-Centric) | $M_1$ (Policy + Env) | $M_2$ (Full Context) | Incremental Lift ($M_2 - M_0$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp 1: Error Reduction** | Mean Absolute Error (MAE) | 5.86 pts | 5.12 pts | **2.68 pts** | **-3.18 pts (-54.3%)** |
| **Exp 1: Error Reduction** | Root Mean Squared Error (RMSE) | 7.42 pts | 6.58 pts | **3.89 pts** | **-3.53 pts (-47.6%)** |
| **Exp 1: Ranking Concordance** | Spearman $\rho$ | 0.482 | 0.591 | **0.874** | **+0.392 (+81.3%)** |
| **Exp 2: Context Sensitivity** | Mean Priority Gap (Susp - Benign) | 0.04 pts | 0.28 pts | **14.22 pts** | **+14.18 pts (Wilcoxon $p < 10^{-15}$)** |
| **Exp 2: Pairwise Accuracy** | Directional Concordance | 50.7% | 53.6% | **92.3%** | **+41.6% (Cohen's $d = 1.42$)** |
| **Exp 3: Triage Retrieval** | NDCG@10 | 0.724 | 0.781 | **0.912** | **+26.0% Lift** |
| **Exp 3: Triage Retrieval** | Recall@25 | 0.433 | 0.533 | **0.800** | **+84.7% Lift** |

---