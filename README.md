# Deep Galerkin Method (DGM) for PDEs

**A mesh-free, computation-conscious implementation of deep learning solvers for high-dimensional partial differential equations.**

Inspired by **Justin Sirignano and Konstantinos Spiliopoulos**, *[DGM: A deep learning algorithm for solving partial differential equations](https://arxiv.org/abs/1708.07469)*, *Journal of Computational Physics* (2018), [DOI: 10.1016/j.jcp.2018.08.029](https://doi.org/10.1016/j.jcp.2018.08.029).

This repository implements experiments based on the paper using **TensorFlow**, with an emphasis on accessible, single-process training instead of the original multi-GPU computing setup. The scripts are adaptations, **not a claim of identical numerical results or hardware-independent accuracy**.

## 1. Core idea

Classical grid-based PDE solvers become expensive as the spatial dimension grows. **DGM replaces the grid with a neural network** $$\(f_\theta\)$$ that represents the solution over the domain. Rather than requiring labeled solutions, it learns directly from the PDE and its conditions by sampling fresh collocation points.

For a parabolic PDE, write

$$
\mathcal{R}u = \partial_t u(t,x) + \mathcal{L}u(t,x) = 0,
\qquad x \in \Omega \subset \mathbb{R}^d,
$$

with prescribed boundary data and initial or terminal data. DGM minimizes the mean-squared errors

$$
\mathcal{J}(\theta) =
\lambda_{\mathrm{PDE}} \, \mathbb{E}_{(t,x)\sim\nu_\Omega}
\left|\mathcal{R} f_\theta\right|^2
+ \lambda_{\mathrm{BC}} \, \mathbb{E}_{(t,x)\sim\nu_{\partial\Omega}}
\left|f_\theta - b\right|^2
+ \lambda_{\mathrm{ICTC}} \, \mathbb{E}_{x\sim\nu_0}
\left|f_\theta(t_*,x) - g(x)\right|^2.
$$

Here \(t_*\) is the initial or terminal time, as appropriate. Derivatives are obtained by **automatic differentiation**; optimization uses **Adam** and independently sampled batches. Thus, training needs no full space-time mesh.

### Model and training algorithm

The paper introduces a **gated, LSTM-like DGM network**, which repeatedly mixes nonlinear transformations of the input and hidden state. This helps approximate PDE solutions with sharp changes or strong nonlinearities.

1. **Sample** interior, boundary, and initial/terminal points (and PDE parameters for a parameterized problem).
2. **Evaluate** the network, its derivatives, and the PDE/condition residuals.
3. **Minimize** the combined loss with a stochastic gradient update.
4. **Resample and repeat**, then compare with analytic/semi-analytic references, numerical benchmarks, or error bounds.

For **American options**, the obstacle \(u\ge g\) introduces an unknown exercise region. The implementation uses the current network approximation to sample continuation/exercise regions and penalizes violations of the payoff constraint. The original paper also proposes a Monte Carlo/antithetic approach to approximating costly second derivatives.

## 2. Numerical experiments

The links below point to the **actual scripts in this repository**. Each result has a reserved image location; upload a figure to the suggested path and replace the placeholder with a Markdown image.

### A. American options — free-boundary PDEs (paper §4)

**Semi-analytic Black–Scholes case — Table 1 / §4.3**  
**Code:** [`black_scholes_semi_analytical_case.py`](black_scholes_semi_analytical_case.py)  
A correlated, **20-dimensional geometric-basket American call** with a known lower-dimensional benchmark. The script currently evaluates \(K=1\), rather than running every dimension in the paper's Table 1.

 <img src="Result/Screenshot 2026-11-23 105313.png"
     alt="Table 1 reference" style="max-width:100%;">


**Lower/upper price bounds — Table 2 / §4.4**  
**Code:** [`american_bound.py`](american_bound.py)  
Uses simulated stock paths to estimate **stopping-based lower and dual-style upper bounds** on the option value. Its bound evaluation uses a **Hutchinson Hessian-trace estimator** to reduce the cost of evaluating the diffusion term. The default case is \(d=20\), \(K=0.95\).

 <img src="Result/Screenshot 2026-11-23 105646.png"
     alt="Table 1 reference" style="max-width:100%;">

### B. Hamilton–Jacobi–Bellman equation (paper §5, Figure 4)

**Code:** [`hjb.py`](hjb.py)  
Solves a **21-dimensional stationary HJB equation** arising from control of a stochastic heat equation. Compares DGM with a quadratic/Riccati reference and plots a spatial **percentage-error contour** (the script saves `figure4_like.png`). Its early-stopping threshold is a *code setting*, not the paper's reported accuracy.

 <img src="Result/Screenshot 2026-11-23 105734.png"
     alt="Table 1 reference" style="max-width:100%;">

### C. Parameterized Burgers' equation (paper §6, Figures 5–7)

The model learns a **family of solutions with one network**, rather than retraining for each choice of viscosity \(\nu\), nonlinearity coefficient \(\alpha\), and boundary values \(a,b\):

$$
\partial_tu+\alpha u\,\partial_xu=\nu\,\partial_{xx}u,\qquad
u(0,x)=a+(b-a)x,\quad u(t,0)=a,\quad u(t,1)=b.
$$

**Figure 5 — Multiple parameter configurations**  
**Code:** [`burger_4_3.py`](burger_4_3.py)  
Compares DGM against finite differences for several choices of \((\nu,\alpha,a,b)\) at \(t=1\).

 <img src="Result/Screenshot 2026-11-23 105823.png"
     alt="Table 1 reference" style="max-width:100%;">

**Figure 6 — Time evolution and viscosity**  
**Code:** [`burger_4_4.py`](burger_4_4.py)  
Studies the formation of steep solution profiles at different times and viscosities, against finite-difference reference curves.

 <img src="Result/Screenshot 2026-11-23 105846.png"
     alt="Table 1 reference" style="max-width:100%;">

**Figure 7 — Error across parameter space**  
**Code:** [`burger_4_5.py`](burger_4_5.py)  
Builds a contour of the **mean absolute error** versus viscosity \(\nu\) and boundary value \(b\), averaged over \(x\) at \(t=1\).

 <img src="Result/Screenshot 2026-11-23 105908.png"
     alt="Table 1 reference" style="max-width:100%;">

## 3. Running the code

```bash
pip install numpy tensorflow matplotlib
python burger_4_3.py                # Burgers: Figure 5
python hjb.py                      # HJB: Figure 4
python black_scholes_semi_analytical_case.py
```

Run the other experiments in the same way using their linked filenames. **Important:** `bs_no_analytic.py` and `american_bound.py` currently load `volatilities_2000_2017(new).csv` and `correlation_matrix_2000_2017(new).csv` from Google Colab Drive paths. **These CSV files are not in this repository**; provide them and update the paths before running those two scripts. Plotting scripts generally display figures interactively; the HJB script saves a PNG.

### Computational notes

- **No spatial mesh for DGM training:** random collocation batches avoid storing a high-dimensional space-time grid.
- **Accessible implementation:** TensorFlow/Adam, local batch processing, and configurable model sizes/iterations remove the paper's six-GPU distributed-training requirement.
- **Trade-offs remain:** the option-training scripts explicitly compute mixed second derivatives, which can be costly as \(d\) grows; the Hutchinson estimator in `american_bound.py` is used for **bound evaluation**, not as a drop-in replacement for those training derivatives.
- **Accuracy must be checked:** compare independent references or option-price bounds after adjusting batch size, training steps, model width, and hardware constraints. Results may differ from the original paper.

## Reference

> Sirignano, J., & Spiliopoulos, K. (2018). **DGM: A deep learning algorithm for solving partial differential equations.** *Journal of Computational Physics*. [Paper](https://arxiv.org/abs/1708.07469) · [DOI](https://doi.org/10.1016/j.jcp.2018.08.029)

This repository is an independent implementation inspired by the research above.
## 4. Historical stock data, EDA, and volatility matrix (2000–2017)

**Reproducible analysis:** [`stock_eda_2000_2017.py`](stock_eda_2000_2017.py) · [Automated generation workflow](.github/workflows/stock_eda.yml)

This additional analysis constructs the heterogeneous-stock inputs used by the American-option experiments. It downloads **all 20 historical equities named in the paper**, including Tiffany (**TIF**) and Nordstrom (**JWN**), from a [frozen copy](https://github.com/shikha-aggarwal/machine_learning/tree/55e3b5b8b47ddd9e93fd3808ac3fff6c85e3b7ce/datasets/stock_data/Stocks) of the [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs). The main archive ends on **November 10, 2017**, not December 31. Two series, **MS** and **TSN**, start in 2005 in that archive; data from [QuantSoftwareToolkit's historical Yahoo snapshot](https://github.com/QuantSoftware/QuantSoftwareToolkit/tree/0eb2c7a776c259a087fdcac1d3ff883eb0b5516c/QSTK/QSData/Yahoo) are rescaled at an overlapping date to backfill their earlier price history. The resulting common window begins in **February 2000**.

### Processing and estimation

1. Download adjusted closing prices; parse dates, remove duplicate/invalid entries, and inspect coverage and missing values.
2. For visualization, normalize each stock's price with a **trailing 60-trading-day z-score**:
   $$
   z_{i,t}^{(60)}=\frac{P_{i,t}-\operatorname{mean}(P_{i,t-59:t})}
   {\operatorname{std}(P_{i,t-59:t})}.
   $$
3. For estimation, use **unnormalized adjusted prices** to calculate daily log returns:
   $$
   r_{i,t}=\log(P_{i,t}/P_{i,t-1}),\qquad
   \sigma_i=\sqrt{252}\,\operatorname{std}(r_i),
   $$
   $$
   \rho_{ij}=\operatorname{corr}(r_i,r_j),\qquad
   \Sigma_{ij}=252\,\operatorname{cov}(r_i,r_j)=\sigma_i\rho_{ij}\sigma_j.
   $$
   All 20 statistics use the same complete-date return observations, without forward-filling.

**Important:** Rolling normalization is for **EDA visualization only**. Computing volatility from the rolling z-scored price series would distort the returns. Here \(\Sigma\) is the **20 × 20 annualized covariance (volatility) matrix**, and \(\sigma\) is the corresponding **20 × 1 annualized volatility vector**.

### Exploratory visualizations

**Rolling-window-normalized prices (60 trading days)**

![60-day normalized historical stock prices](results/prices_60d_zscore.svg)

**Annualized volatility by stock**

![Annualized volatility of all 20 equities](results/annualized_volatility.svg)

**20 × 20 annualized covariance / volatility matrix**

![Annualized covariance matrix heatmap](results/annualized_covariance_matrix.svg)

**20 × 20 correlation matrix**

![Daily log-return correlation matrix](results/correlation_matrix.svg)

**Sample EDA findings.** Across **4,451 aligned daily returns**, Amazon (**AMZN**) has the highest annualized volatility (**52.97%**), followed by Morgan Stanley (**MS**, **50.18%**) and Bank of America (**BAC**, **47.31%**). General Mills (**GIS**, **18.21%**) and Johnson & Johnson (**JNJ**, **18.78%**) show lower volatility. The strongest pairwise sample correlation is **BAC–MS: 0.649**. These are **independent historical estimates**, not the original paper's quoted results.

### Reproduce and use the output

```bash
pip install numpy pandas matplotlib
python stock_eda_2000_2017.py
```

The script saves:

- [`data/volatilities_2000_2017.csv`](data/volatilities_2000_2017.csv) — annualized \(\sigma_i\), one number per line in the ticker order defined in the script.
- [`data/correlation_matrix_2000_2017.csv`](data/correlation_matrix_2000_2017.csv) — 20 × 20 matrix \(\rho\).
- [`data/annualized_covariance_matrix_2000_2017.csv`](data/annualized_covariance_matrix_2000_2017.csv) — 20 × 20 matrix \(\Sigma\).
- [`data/data_quality.csv`](data/data_quality.csv) and [`data/eda_summary.csv`](data/eda_summary.csv) — coverage and descriptive statistics.

To use the new files with [`bs_no_analytic.py`](bs_no_analytic.py) or [`american_bound.py`](american_bound.py), replace their **Google Colab Drive paths** for the two input CSVs with `data/volatilities_2000_2017.csv` and `data/correlation_matrix_2000_2017.csv`. This is an independently reconstructed historical calibration, **not a claim of exact equality to the original paper's proprietary data preparation**.
