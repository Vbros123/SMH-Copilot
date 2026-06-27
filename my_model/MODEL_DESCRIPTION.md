# Model Description — Round 20 COVID-19 Scenario Modeling Hub

## Overview

This prototype submission uses a **stochastic log-growth random-walk model
with dual-harmonic seasonal forcing** to produce 300 paired trajectories
per scenario.  The model is framed as a **discrete-time stochastic
difference equation** operating in log-space, which is mathematically
equivalent to a geometric Brownian motion (GBM) sampled at weekly intervals.

---

## Model Type

| Property | Value |
|----------|-------|
| Framework | Stochastic discrete-time difference equation (log-space GBM) |
| Compartments | None (reduced-form / semi-mechanistic) |
| ODE system | See §3 for the equivalent continuous-time ODE |
| Observations fitted | Weekly incident hospitalisations (`inc hosp`) |
| Deaths derived | Proportional to hospitalisations via HOSP_TO_DEATH_RATIO = 6.5% |
| Trajectories | 300 independent samples per scenario |
| Projection horizon | 104 epi-weeks (Jun 2025 – Jun 2027) |

---

## 1  State Variables

The model tracks a single observable state:

$$
H(t) = \text{weekly incident hospitalisations in epi-week } t
$$

All computation is performed in log-space:

$$
h(t) = \ln H(t)
$$

---

## 2  Stochastic Difference Equation

### 2.1  Core recursion

For trajectory $i$ and week $t$:

$$
h_i(t) = h_i(t-1) + \mu^* + s(t) + \varepsilon_i(t)
$$

where:

| Symbol | Definition |
|--------|-----------|
| $h_i(t)$ | $\ln H_i(t)$, log-hospitalisation for trajectory $i$ at week $t$ |
| $\mu^*$ | Damped trend drift: $\mu^* = (1 - \alpha_r)\,\hat\mu$ |
| $\hat\mu$ | Estimated mean log-growth rate from the 26-week calibration window |
| $\alpha_r$ | Mean-reversion strength = 0.15 (prevents divergence over 104 weeks) |
| $s(t)$ | Dual-harmonic seasonal forcing (see §2.2) |
| $\varepsilon_i(t)$ | Innovation: $\varepsilon_i(t) \overset{iid}{\sim} \mathcal{N}(0,\,\hat\sigma^2)$ |

The observation in natural units is recovered as:

$$
H_i(t) = \max\!\bigl(\exp(h_i(t)),\; H_{\min}\bigr), \quad H_{\min} = 200 \text{ (US national floor)}
$$

### 2.2  Dual-harmonic seasonal forcing

COVID-19 hospitalisations exhibit **two seasonal peaks per year**: a primary
winter peak (≈ January 18) and a secondary summer peak (≈ July 15).  The
seasonal component is the week-to-week derivative of the sum of two
sinusoids in log-count space:

$$
S(t) = A_1 \sin\!\Bigl(\omega(d(t) - d_1)\Bigr)
     + A_2 \sin\!\Bigl(\omega(d(t) - d_2)\Bigr)
$$

The seasonal component in log-count space uses a **Fourier dual-harmonic** form
fitted to 2023–2026 US NHSN weekly data by OLS:

$$
S(d) = \underbrace{a_{1c}\cos(\omega d) + a_{1s}\sin(\omega d)}_{\text{annual harmonic}}
     + \underbrace{a_{2c}\cos(2\omega d) + a_{2s}\sin(2\omega d)}_{\text{semi-annual harmonic}}
$$

where $d$ = day-of-year.  The annual harmonic captures the dominant winter peak
(≈ early January); the semi-annual harmonic captures the secondary summer peak
(≈ August/September), producing the **two peaks per year** required by Round 20.

**The trajectory generator uses $S(d)$ directly (level model), not its derivative.**
This avoids the cumulative-drift collapse that occurs when the derivative
approach is applied over a 104-week horizon starting at a seasonal trough.

| Coefficient | Value | Description |
|-------------|-------|-------------|
| $\omega$ | $2\pi/365.25$ rad/day | Annual angular frequency |
| $a_{1c}$ | 0.492 | Annual cosine coefficient (fitted OLS, 2023–2026) |
| $a_{1s}$ | −0.026 | Annual sine coefficient |
| $a_{2c}$ | −0.006 | Semi-annual cosine coefficient |
| $a_{2s}$ | 0.436 | Semi-annual sine coefficient |

The drift $s(t)$ in the ODE representation is the derivative of $S$ (×7 for weekly units),
but is not used directly in the discrete trajectory generator.

The derivative form is used so that $H(t)$ follows the *shape* of the
double-sinusoidal seasonal curve — both the winter 2026-27 peak and the
summer 2026 peak emerge naturally from this forcing.

---

## 3  Equivalent Continuous-Time ODE

The discrete recursion in §2.1 corresponds to the following
**stochastic differential equation (SDE)** in continuous time:

$$
\frac{dh}{dt} = \mu^* + s(t) + \sigma\,\frac{dW}{dt}
$$

where $W(t)$ is a standard Wiener process (Brownian motion).  In the
epidemic literature this is an **additive noise** model for the
log-epidemic trajectory.

Equivalently, in terms of the count $H = e^h$:

$$
\frac{dH}{dt} = H(t)\!\left[\mu^* + s(t) + \tfrac{1}{2}\sigma^2\right] + H(t)\,\sigma\,\frac{dW}{dt}
$$

This is a **geometric Brownian motion with seasonal drift**, the simplest
continuous-time ODE consistent with exponential epidemic growth/decay and
log-normal forecast uncertainty.

> **Note on recursion and ODE equivalence.**  The discrete recursion
> $h_i(t) = h_i(t-1) + \mu^* + s(t) + \varepsilon_i(t)$ is *not* circular;
> it is a first-order Markov process — each step depends only on the
> immediately preceding value.  The ODE form above is the formal
> continuous-time limit (Itô convention).  Numerically, the discrete-time
> version is used because the hub submissions require weekly counts.

---

## 4  Vaccination Adjustment (Scenario Multiplier)

Scenarios B–E modify the baseline trajectories by a **deterministic
per-week multiplier** $M_k(t)$ that represents the reduction in
susceptibility due to vaccination under scenario $k$:

$$
H_k(t) = H_A(t) \times M_k(t), \quad M_k(t) \in [0.5,\, 1.0]
$$

### 4.1  Population-level protection

The multiplier is derived from the **convolution** of the weekly
vaccination rate $\Delta V(s)$ with the time-varying effective vaccine
effectiveness $\mathrm{VE}(t-s)$:

$$
P(t) = \sum_{s \leq t} \Delta V(s) \cdot \mathrm{VE}(t - s) \cdot w
$$

$$
M_k(t) = 1 - P_{\text{fall}}(t) - P_{\text{spring}}(t)
$$

where $w$ is a population weight (1.0 for whole-population campaigns,
$f_{\text{HR}} = 0.30$ for high-risk-only spring campaigns).

### 4.2  Effective VE with waning and immune escape

$$
\mathrm{VE}(\tau) = \mathrm{VE}_0
  \underbrace{\Bigl[\varphi + (1-\varphi)\,e^{-\lambda\tau}\Bigr]}_{\text{waning factor}}
  \underbrace{(1 - \kappa)^{\tau/52}}_{\text{immune escape}}
$$

| Symbol | Value | Description |
|--------|-------|-------------|
| $\mathrm{VE}_0$ | 0.55 | Initial VE against hospitalisation (round20.md) |
| $\varphi$ | 0.50 | Waned plateau fraction (residual VE / VE_0) |
| $\lambda$ | $\ln 2 / 26$ week$^{-1}$ | Exponential waning rate (half-life = 26 weeks) |
| $\tau$ | — | Weeks since vaccination |
| $\kappa$ | 0.35 yr$^{-1}$ | Annual immune escape fraction (midpoint 20–50%) |

### 4.3  Scenario specifications

| Scenario | Fall 2026-27 | Spring 2026 (HR only) | Fall coverage | Spring coverage |
|----------|-------------|----------------------|---------------|-----------------|
| A | ✗ | ✗ | 0% | 0% |
| B | ✓ (BaU) | ✗ | 33% | 0% |
| C | ✓ (BaU) | ✓ | 33% | 16.5% |
| D | ✓ (Opt) | ✗ | 59% | 0% |
| E | ✓ (Opt) | ✓ | 59% | 29.5% |

---

## 5  Death Derivation

Incident deaths are derived from hospitalisation trajectories using a fixed
case-hospitalisation-fatality ratio:

$$
D(t) = H(t) \times r_{\text{HD}}, \quad r_{\text{HD}} = 0.065 \ (6.5\%)
$$

This is a simplification; a full model would run independent death
trajectories seeded from infection counts.

---

## 6  Parameter Estimation

| Parameter | Method | Window |
|-----------|--------|--------|
| $\hat\mu$ (drift) | Sample mean of log-differences | 52 most-recent weeks (full seasonal cycle) |
| $\hat\sigma$ (volatility) | Sample std of log-differences (ddof=1) | 52 most-recent weeks |
| $A_1, A_2, d_1, d_2$ | Fixed; calibrated from US national 2023–2026 data | Full history |
| $\alpha_r$ (mean reversion) | Fixed; 0.15 | — |

---

## 7  Key Assumptions

| # | Assumption | Justification |
|---|-----------|--------------|
| A1 | Log-growth rates are stationary over 52 weeks | One full seasonal year; mean ≈ 0 for endemic seasonal pathogen |
| A2 | Innovations are i.i.d. Normal | Parsimony; tail risk partially captured by seasonal forcing |
| A3 | Dual-harmonic sinusoidal seasonality | Observed summer + winter COVID-19 peaks in 2023–2026 NHSN data |
| A4 | Geometric Brownian motion on counts | Ensures non-negativity; percentage uncertainty grows with horizon |
| A5 | Trajectories paired across scenarios via shared seed | Required by round20.md pairing convention |
| A6 | Endemic floor $H_{\min} = 200$ nationally | COVID-19 remains endemic; NHSN data never below 100/week since 2020 |

---

## 8  Collaboration / Multi-State Extension

For state-level submissions, the same model structure is applied
independently to each state using state-specific observation series.
The scenario multipliers use **national** vaccination coverage curves
from `auxiliary-data/vaccination-coverage/COVID_RD20_Vaccination_curves.csv`
as a proxy; teams with state-level coverage data should substitute those
directly into `build_protection_schedule()`.

The model is **not** a metapopulation model — state trajectories are
generated independently.  Spatial coupling (travel, commuting) is not
modelled in this prototype.

---

## 9  File Structure

```
my_model/
├── MODEL_DESCRIPTION.md       ← this file
├── simulate.py                ← entry point (builds submission end-to-end)
├── stochastic_simulate.py     ← §2 dual-harmonic GBM trajectory generator
├── scenario_adjustments.py    ← §4 vaccination multiplier / VE model
├── build_submission.py        ← §5 hub parquet assembly & validation
├── plot_submission.py         ← visualization (US + state-level charts)
└── load_data.py               ← data loading (NHSN, locations, vax curves)
```

---

## 10  References

- Round 20 scenario specification: `auxiliary-data/rounds/round20.md`
- Submission format: `model-output/README.md`
- Vaccination coverage curves: `auxiliary-data/vaccination-coverage/`
- CDC NHSN weekly hospitalisations: `target-data/time-series.csv`
- Oksendal, B. (2003). *Stochastic Differential Equations*, 6th ed. Springer.
- Black, F. & Scholes, M. (1973). The pricing of options. *J. Political Economy.*
  (GBM framework basis)
