# SymTorch Benchmark — Strategy & Guide

## What we're doing

[SymTorch](https://github.com/astroautomata/SymTorch) (`pip install torch-symbolic`) wraps any PyTorch component, records its input/output activations, then runs **PySR** (genetic symbolic regression) to recover a human-readable equation. The idea is that if a NN learned `sin(x)`, the weights encode that — SymTorch reverse-engineers it.

Our benchmarks follow a clean protocol:

```
1. Sample input domain (e.g., x ∈ [-3, 3])
2. Compute ground-truth outputs from the analytic formula
3. Add Gaussian noise to simulate real-world training data
4. Train a small MLP (tanh activations, 3-4 layers, 64-128 neurons)
5. Call SymTorch to distill the trained MLP → symbolic equation
6. Compare: ground truth vs. NN output vs. symbolic output
   using MSE and R²
```

---

## Benchmark Suite (6 functions)

| # | Function | Truth | Difficulty | Notes |
|---|----------|-------|------------|-------|
| 1 | `x²` | `x²` | Easy | Baseline, clean data, should recover exactly |
| 2 | `sin(x) + noise` | `sin(x)` | Medium | Noisy, transcendental, σ_noise=0.1 |
| 3 | `x·exp(-x²)` | `x·exp(-x²)` | Medium | Mixed polynomial + exponential |
| 4 | Normal PDF `φ(x)` | `0.3989·exp(-0.5x²)` | Hard | Very small values near tails |
| 5 | Black-Scholes call | `S·Φ(d₁) - Ke^{-rT}·Φ(d₂)` | Very Hard | 2D, involves CDF, open-ended |
| 6 | `x·y` | `x·y` | Easy-Medium | Multi-input interaction test |

---

## Why these functions?

**Quadratic (`x²`)**: PySR should nail this immediately. If it doesn't, something's wrong with your install or the NN didn't converge.

**`sin(x)` + noise**: Tests whether SymTorch can recover transcendental functions through noise. PySR has sin/cos in its operator set by default.

**`x·exp(-x²)`**: This is the derivative of a Gaussian, which appears everywhere (physics, finance, ML kernels). Requires PySR to compose multiplication with exp.

**Normal PDF**: A real challenge — the constant `1/√(2π) ≈ 0.3989` needs to be discovered, and the `exp(-x²/2)` must emerge. Good test of constant discovery.

**Black-Scholes**: The holy grail test. The true formula involves the normal CDF `Φ`, which PySR can't directly express unless you add it as a custom operator. What you'll likely get is a polynomial/rational approximation of the BS surface — which is actually very interesting! If PySR recovers something like `σ·√T · 0.4 · S` for ATM options, that's the well-known approximation `C ≈ 0.4·S·σ·√T`.

**`x·y`**: Tests cross-term interaction. Neural networks struggle to learn pure multiplication efficiently, so this tests whether SymTorch can see through the approximation.

---

## Installation

```bash
pip install torch-symbolic pysr torch numpy scipy matplotlib
```

PySR has a Julia backend. On first run it will auto-install Julia — this takes a few minutes but only happens once.

```bash
python symtorch_benchmark.py
```

---

## SymTorch API Quick Reference

```python
from symtorch import SymbolicModel

# --- Model-agnostic mode (any callable, any framework) ---
def my_model(x: np.ndarray) -> np.ndarray:
    ...  # your PyTorch/sklearn/whatever model

sym = SymbolicModel(my_model)
sym.distill(
    X_train,                           # numpy array (N, n_features)
    fit_params={"variable_names": ["x", "y"]},
    sr_params={"niterations": 40, "verbosity": 0},
)
print(sym.equations_)  # DataFrame with equation, loss, complexity columns

# --- Layer-level mode (wrap a specific nn.Module layer) ---
from symtorch import SymbolicModel
import torch.nn as nn

model = ...  # your full model
model.layer1 = SymbolicModel(model.layer1)  # wrap the layer

# Run a forward pass with sample data to collect I/O
model(sample_data)

# Then distill
model.layer1.distill(fit_params={"variable_names": [...]})

# Optionally replace layer with the symbolic equation in forward pass
model.layer1.use_equation(idx=0)  # use Pareto-front equation #0
```

---

## What to look for in results

- **Best case**: PySR recovers the exact symbolic form (e.g., `x²`, `sin(x)`)
- **Expected case**: PySR finds a close approximation (e.g., `0.3989 * exp(-0.4999 * x²)` for Normal PDF)
- **Hard case**: Black-Scholes will likely give a polynomial surface fit over [σ, T] space — compare it to the known ATM approximation `0.4 · S · σ · √T`
- **Check the `equations_` DataFrame**: it has the full Pareto front (complexity vs. accuracy tradeoff). Lower complexity = more interpretable; lower loss = more accurate. The "best" equation balances both.

---

## Extensions to try

1. **SLIME mode**: local interpretability around a specific point
   ```python
   sym.distill(X, SLIME=True, slime_params={"center": x0, "n_samples": 500})
   ```

2. **Custom operators for Black-Scholes**: add `norm.cdf` as a PySR operator
   ```python
   from scipy.stats import norm
   sr_params = {
       "extra_sympy_mappings": {"Phi": norm.cdf},
       "custom_operators": ["Phi(x) = normcdf(x)"],
   }
   ```

3. **Replace layer in forward pass**: after distillation, swap the NN layer for the equation and measure speed/accuracy tradeoff.

4. **Riemann zeta**: requires complex input, but you can approximate `ζ(s)` on the real line `s > 1` using the MLP → SymTorch pipeline. Try `s ∈ [1.5, 10]`.

5. **Famous pretrained models**: wrap a ResNet MLP layer or a transformer MLP and see what equations emerge — this is what the paper actually demonstrates on GPT-2.
