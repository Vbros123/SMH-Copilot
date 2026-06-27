# Model Description — Round 20 COVID-19 Scenario Modeling Hub

**Team:** MyTeam-ProtoModel  
**Round:** 20  
**Projection period:** Jun 8, 2025 → Jun 5, 2027 (104 epi-weeks)  
**Calibration period:** Jan 1, 2024 → Jun 6, 2026  

---

## Overview

This submission uses a **stochastic SEIRS compartmental model** with seasonal
transmission forcing, waning immunity, and vaccination-mediated susceptibility
reduction.  The model is implemented in discrete weekly time steps and calibrated
independently for each location against observed NHSN weekly hospitalisations.

---

## Compartments

| Symbol | Name | Description |
|--------|------|-------------|
| S | Susceptible | No current immunity; can be infected |
| E | Exposed | Infected but not yet infectious (latent period) |
| I | Infectious | Actively infectious; drives transmission |
| R | Recovered | Post-infectious immunity; wanes back to S |

Observable output (not a compartment):

$$
H(t) = \frac{I(t)}{D_I} \cdot p_{\text{hosp}} \cdot N
$$

where $H(t)$ is weekly incident hospitalisations, $D_I$ is the mean infectious
period, $p_{\text{hosp}}$ is the infection-hospitalisation rate, and $N$ is the
population.

---

## ODE / Difference Equations

The discrete-time weekly system is:

$$
\lambda(t) = \frac{\beta(t) \cdot I(t)}{N}
$$

$$
\Delta E(t)  = S(t) \cdot \lambda(t) \quad \text{(new exposures — stochastic, see below)}
$$

$$
\Delta I(t)  = \frac{E(t)}{D_E}  \quad \text{(latency progression)}
$$

$$
\Delta R(t)  = \frac{I(t)}{D_I}  \quad \text{(recovery)}
$$

$$
\Delta S(t)  = R(t) \cdot \omega  \quad \text{(waning immunity)}
$$

where $\omega = 1 - e^{-1/\tau_R}$ is the weekly waning rate.

Compartment updates:

$$
S(t+1) = S(t) - \Delta E(t) + \Delta S(t)
$$
$$
E(t+1) = E(t) + \Delta E(t) - \Delta I(t)
$$
$$
I(t+1) = I(t) + \Delta I(t) - \Delta R(t)
$$
$$
R(t+1) = R(t) + \Delta R(t) - \Delta S(t)
$$

---

## Seasonality Equation

Transmission rate is modulated by a seasonal cosine forcing:

$$
\beta(t) = \beta_0 \left(1 + A \cos\!\left(\frac{2\pi(t - \varphi)}{52}\right)\right)
$$

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Baseline transmission | $\beta_0$ | calibrated | Weekly transmission rate (week⁻¹) |
| Seasonal amplitude | $A$ | calibrated | Fraction of variation (0–1) |
| Phase offset | $\varphi$ | calibrated | Epi-week of seasonal peak (1–52) |

**Physical interpretation:** $\beta(t)$ oscillates between $\beta_0(1-A)$ in the
trough and $\beta_0(1+A)$ at the peak.  For endemic stability, $\beta_0 \cdot D_I > 1$
(i.e., $R_0 > 1$), ensuring the endemic equilibrium exists.  The seasonal forcing
drives annual epidemic waves around this endemic state.

---

## Noise Model

New exposures are drawn from a **negative-binomial distribution** to capture
overdispersed transmission (super-spreading):

$$
\Delta E(t) \sim \text{NegBin}\!\left(\mu = S(t)\lambda(t),\; \varepsilon = 0.10\right)
$$

where $\varepsilon$ is the overdispersion parameter (variance = $\mu + \varepsilon \mu^2$).
Setting $\varepsilon = 0$ reduces to Poisson noise.  The value $\varepsilon = 0.10$
is consistent with COVID-19 transmission cluster data (Lloyd-Smith et al. 2005).

---

## Vaccination Implementation

Vaccination is modelled as a **per-week multiplicative reduction** in effective
susceptibility:

$$
S_{\text{eff}}(t) = S(t) \cdot M_k(t)
$$

where $M_k(t) \in [0.5, 1.0]$ is the scenario-specific protection multiplier for
scenario $k$ at week $t$.

The multiplier is derived from the convolution of weekly vaccination rates with
time-varying effective vaccine effectiveness:

$$
P(t) = \sum_{s \leq t} \Delta V(s) \cdot \mathrm{VE}(t - s)
$$

$$
M_k(t) = 1 - P_{\text{fall}}(t) - P_{\text{spring}}(t)
$$

Effective VE with waning and immune escape:

$$
\mathrm{VE}(\tau) = \mathrm{VE}_0
  \underbrace{\left[\varphi_{\text{wane}} + (1-\varphi_{\text{wane}})\,e^{-\lambda\tau}\right]}_{\text{waning}}
  \underbrace{(1 - \kappa)^{\tau/52}}_{\text{immune escape}}
$$

| Symbol | Value | Description |
|--------|-------|-------------|
| $\mathrm{VE}_0$ | 0.55 | Initial VE against hospitalisation (round20.md) |
| $\varphi_{\text{wane}}$ | 0.50 | Residual VE fraction at waned plateau |
| $\lambda$ | $\ln 2 / 26$ week⁻¹ | Exponential waning rate (half-life = 26 weeks) |
| $\kappa$ | 0.35 yr⁻¹ | Annual immune escape fraction |

---

## Transition Rates and Parameter Definitions

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Mean latent period | $D_E$ | 1.5 weeks | He et al. Nat Med 2020 |
| Mean infectious period | $D_I$ | 1.5 weeks | Cevik et al. Lancet Microbe 2021 |
| Waning timescale | $\tau_R$ | 78 weeks | Levin et al. Nat Commun 2022 |
| NB overdispersion | $\varepsilon$ | 0.10 | Lloyd-Smith et al. Nature 2005 |
| IHR multiplier | $p_{\text{hosp}}$ | calibrated | NHSN 2024-26 |
| In-hospital CFR | $r_{\text{HD}}$ | 0.065 | CDC NCHS 2025-26 |
| Initial recovered fraction | $f_{R,0}$ | 0.33 | Near endemic equilibrium |

---

## Calibration Method

Parameters are fitted by **Nelder-Mead minimisation** of the sum of squared
log-residuals between simulated and observed weekly hospitalisations:

$$
\mathcal{L}(\theta) = \sum_{t: H^{\text{obs}}_t > 0} \left(\ln H^{\text{sim}}_t(\theta) - \ln H^{\text{obs}}_t\right)^2
$$

A **stability penalty** is added to prevent runaway growth:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fit}} + \alpha \cdot \max\!\left(0,\, \frac{\max H^{\text{sim}}_{\text{future}}}{2 \cdot \max H^{\text{obs}}} - 1\right)^2
$$

Optimised parameters per location: $\{\beta_0, A, \varphi, p_{\text{hosp}}, \log_{10}(I_0/N)\}$

Parameter bounds enforced during calibration:

| Parameter | Lower | Upper | Rationale |
|-----------|-------|-------|-----------|
| $\beta_0$ | 0.68 | 6.3 | Ensures $R_0 = \beta_0 D_I > 1$ (endemic condition) |
| $A$ | 0.05 | 0.85 | Non-trivial seasonality |
| $\varphi$ | 0 | 52 | Any week of year |
| $p_{\text{hosp}}$ | 0.03% | 3% | COVID-19 IHR literature range |
| $\log_{10}(I_0/N)$ | −4.0 | −0.5 | Initial infectious fraction |

---

## State-Specific Fitting

Each location (US national + 50 states + territories) is calibrated independently
using its own observed hospitalisation series from `target-data/time-series.csv`.
Population data is loaded from `auxiliary-data/data-locations/locations.csv`.

State-specific calibration allows the model to capture differences in:
- **Baseline transmission** $\beta_0$ (population density, behaviour)
- **Seasonal amplitude** $A$ (climate, indoor crowding patterns)
- **Seasonal phase** $\varphi$ (timing of winter vs. summer waves by latitude)
- **IHR** $p_{\text{hosp}}$ (age structure, healthcare access, vaccination rates)
- **Initial epidemic state** $I_0$ (current infection burden)

---

## Seeding Strategy (Calibration vs. Forecast)

The separation of calibration and forecast periods is critical for correct
temporal interpretation:

1. **Calibration** (`simulate.calibrate_seirs`): Fit SEIRS parameters against
   NHSN data from Jan 2024 → Jun 2026 using deterministic (noise-free) dynamics.

2. **Warm-up** (`stochastic_simulate.generate_trajectories`): Run the
   deterministic SEIRS from `CAL_START` to `ORIGIN_DATE` (Jun 8, 2025) to
   determine the compartment state at the forecast start.

3. **Projection** (`simulate.generate_seirs_trajectories`): Generate 300
   stochastic trajectories from the warm-up state through 104 forecast weeks
   (Jun 2025 → Jun 2027).

This design ensures the forecast begins at the epidemiologically correct state,
and the calibration period is never mixed with the forecast period on plots.

---

## Scenario Implementation

Scenarios A–E modify only the **vaccination protection multiplier** $M_k(t)$
applied to effective susceptibles.  The underlying SEIRS parameters
($\beta_0, A, \varphi, D_E, D_I, \tau_R$) are **identical across all scenarios**.

| Scenario | Description | Fall 2026-27 campaign | Spring 2026 HR campaign |
|----------|-------------|----------------------|-------------------------|
| A | No further vaccination | ✗ | ✗ |
| B | BaU annual coverage | ✓ (33%) | ✗ |
| C | BaU + spring HR | ✓ (33%) | ✓ (16.5%) |
| D | Optimistic annual | ✓ (59%) | ✗ |
| E | Optimistic + spring HR | ✓ (59%) | ✓ (29.5%) |

Scenario ordering: A > B > C > D > E (decreasing hospitalisations with more vaccination).

---

## Known Limitations

1. **Calibration quality**: The Nelder-Mead optimiser finds a local minimum.
   The calibration fit underestimates the observed peak magnitude in some
   locations.  A global optimisation method (differential evolution, MCMC)
   would improve parameter recovery.

2. **Single-age model**: All compartments are age-homogeneous.  Age-stratified
   IHR and vaccination coverage vary substantially; this simplification
   underestimates scenario heterogeneity.

3. **Deaths derived**: Deaths are computed as $H(t) \times r_{\text{HD}} = 0.065$
   rather than from an independent death compartment.  This ignores variation
   in case fatality by age and variant.

4. **Independent state simulations**: States are modelled independently with
   no spatial coupling.  Interstate travel and commuting are not modelled.

5. **Fixed structural parameters**: $D_E$, $D_I$, $\tau_R$, and $\varepsilon$
   are fixed at literature-based national defaults and not fitted per location.
   State-level variation in these quantities is ignored.

6. **Immune escape**: Modelled only in the vaccination multiplier, not in the
   SEIRS transmission dynamics.  Antigenic drift could change $\beta_0$ over
   time (not captured).

7. **Variant emergence**: No mechanism for sudden jumps in transmissibility
   from new variant emergence.

---

## File Structure

```
my_model/
├── MODEL_DESCRIPTION.md         ← this file
├── simulate.py                  ← SEIRS model + calibration (core)
├── stochastic_simulate.py       ← public API (backward-compatible interface)
├── scenario_adjustments.py      ← vaccination multiplier (Scenarios A–E)
├── build_submission.py          ← hub parquet assembly & validation
├── plot_submission.py           ← publication-quality visualisation
└── load_data.py                 ← data loading (NHSN, locations, vax curves)
```

---

## References

1. He X, et al. (2020). Temporal dynamics in viral shedding and transmissibility of COVID-19. *Nature Medicine* 26:672–675.
2. Cevik M, et al. (2021). SARS-CoV-2, SARS-CoV, and MERS-CoV viral load and shedding kinetics. *Lancet Microbe* 2:e13-e22.
3. Levin AT, et al. (2022). Assessing the burden of COVID-19 in developing countries. *Nature Communications* 13:1–10.
4. Lloyd-Smith JO, et al. (2005). Superspreading and the effect of individual variation on disease emergence. *Nature* 438:355–359.
5. Kissler SM, et al. (2020). Projecting the transmission dynamics of SARS-CoV-2 through the postpandemic period. *Science* 368:860–868.
6. CDC NHSN COVID-19 Hospitalization Data. https://www.cdc.gov/nhsn/covid19/
7. Round 20 scenario specification: `auxiliary-data/rounds/round20.md`
