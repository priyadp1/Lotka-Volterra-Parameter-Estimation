# Lotka-Volterra Parameter Estimation

Estimates parameters of the Lotka-Volterra predator-prey model using three different approaches — nonlinear least squares (Python), Euler integration with fixed parameters (Python), and Hamiltonian-based random search (Java) — all fit to historical hare and lynx population data (1847–1903).

## The Model

```
dx/dt = α·x − β·x·y   (prey: hare)
dy/dt = δ·x·y − γ·y   (predator: lynx)
```

| Parameter | Description |
|-----------|-------------|
| α (a) | Hare intrinsic growth rate |
| β (b) | Predation coefficient |
| δ (c) | Predator growth efficiency from predation |
| γ (d) | Lynx natural death rate |

## Data

`Leigh1968_harelynx.csv` — 57 years of Canadian hare and lynx fur-trading records (Leigh, 1968), spanning 1847–1903. Populations are scaled to thousands of animals.

## Files

| File | Language | Description |
|------|----------|-------------|
| `NLS_code.py` | Python | Nonlinear least squares parameter estimation using `scipy.optimize.minimize` (L-BFGS-B) and `solve_ivp` ODE solver |
| `plot_hare_lynx.py` | Python | Euler method simulation using pre-fit parameters; plots time series and phase portraits |
| `phase_portrait_ham.py` | Python | Euler method simulation using Hamiltonian-derived parameters; plots phase portrait comparison |
| `HamiltonianEstimator.java` | Java | Hamiltonian-based parameter estimation via random search with iterative refinement |

## Methods

### 1. Nonlinear Least Squares (`NLS_code.py`)
Minimizes the sum of squared errors between the ODE solution and observed data. Uses `scipy.integrate.solve_ivp` to numerically solve the equations and `scipy.optimize.minimize` with L-BFGS-B and bounded parameters.

**Output:** `lotka_volterra_fit.png`, `lotka_volterra_phase_portrait.png`

### 2. Euler Integration (`plot_hare_lynx.py`)
Simulates the model using a fixed Euler step (`h = 0.01`) and pre-estimated parameters (α=1.595, β=0.066, δ=2.047, γ=0.063). Plots observed data vs. model predictions.

**Output:** `model_phase_portrait.png`, `observed_vs_model_phase_portrait.png`

### 3. Hamiltonian Phase Portrait (`phase_portrait_ham.py`)
Uses the Lotka-Volterra Hamiltonian (conserved quantity) with Euler integration to draw the model phase portrait and overlay it against observed data.

**Output:** `phase_portrait_comparison.png`

### 4. Hamiltonian Estimator (`HamiltonianEstimator.java`)
Estimates parameters by minimizing a weighted objective that penalizes:
- Variance of the Hamiltonian H = d·x − c·ln(x) + b·y − a·ln(y) (should be conserved)
- Deviation of the equilibrium point (x\*, y\*) from the data mean
- Deviation of the implied oscillation period from the observed period

Uses random search with iterative refinement (5 rounds, 4000 samples/round, 35% shrinkage).

## Usage

### Python scripts

**Install dependencies:**
```bash
pip install numpy pandas matplotlib scipy
```

**Run NLS estimation:**
```bash
python3 NLS_code.py
```

**Run Euler simulation and plots:**
```bash
python3 plot_hare_lynx.py
python3 phase_portrait_ham.py
```

### Java Hamiltonian Estimator

**Compile and run:**
```bash
javac HamiltonianEstimator.java
java HamiltonianEstimator
```

> Note: expects the data file to be named `Leigh1968_harelynx.csv` in the working directory.

## Output Files

| File | Description |
|------|-------------|
| `lotka_volterra_fit.png` | NLS model vs. observed time series |
| `lotka_volterra_phase_portrait.png` | NLS phase portrait overlay |
| `model_phase_portrait.png` | Euler model-only phase portrait |
| `observed_vs_model_phase_portrait.png` | Euler observed vs. model phase portrait |
| `phase_portrait_comparison.png` | Hamiltonian model vs. observed phase portrait |

## License

MIT License — Copyright (c) 2026 Prisha Priyadarshini
