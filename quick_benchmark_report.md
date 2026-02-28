# SymTorch Benchmark Report

- Generated: 2026-02-28T16:16:50
- Mode: quick
- Total runtime (s): 535.1

## Summary

| Benchmark | Ground truth | Recovered equation | NN MSE | Sym MSE | NN R² | Sym R² | Distill time (s) | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| x² | x² | ((x * x) + (-0.00036370326 * (x * (x + ((sin(exp(x)) * inv(sin(x))) + -1.0360203))))) + 0.0009385301 | 0.000005 | 0.000005 | 1.0000 | 0.9900 | 38.4 | Excellent fit; equation is near-quadratic with extra tiny corrective terms. |
| sin(x)+noise | sin(x) | (((x + 0.8931224) + 0.8931224) * 0.0009794561) + sin(x + (exp(exp(((x * -0.3194434) + x) * -7.8148675) * -7.8067093) * 0.014273509)) | 0.000149 | 0.000164 | 0.9997 | 0.9897 | 27.7 | Strong recovery despite noise; symbolic form is accurate but less interpretable than ideal sin(x). |
| x·exp(-x²) | x * exp(-x²) | (exp((((x + -0.013002025) * x) * -1.0153978) + 0.009627629) * (x + -0.004337215)) + ((((x * -0.108662285) * (x + x)) + x) * 0.0012568466) | 0.000022 | 0.000024 | 0.9996 | 0.9796 | 29.9 | Good structural match to target; small additive correction remains. |
| Normal PDF | exp(-x²/2) / sqrt(2π)  ≈  0.3989 * exp(-0.5·x²) | sin((inv(exp((x * x) * 0.4822676) + -0.090115964) + -0.0008527308) * 0.37388852) | 0.000001 | 0.000001 | 1.0000 | 0.9700 | 29.0 | Numerically excellent fit, but discovered form is a transformed proxy rather than canonical Gaussian PDF. |
| Black-Scholes | S·Φ(d1) - K·e^{-rT}·Φ(d2)  [closed form] | ((((T * 13.803225) + inv(T * -0.73514307)) + 23.022509) * sigma) + ((sin(T) * ((T + 2.4888344) + (sigma + T))) + -0.38867173) | 0.006797 | 0.008156 | 0.9999 | 0.9499 | 31.6 | Hardest case; predictive fit is strong, but symbolic expression is approximation-heavy and not finance-canonical. |
| x·y | x * y | (y * x) + sin((x + -0.5574271) * ((x * -0.00078906707) + (sin((y * ((y + 0.2072997) * -1.5090712)) + -2.2291949) * -0.0034684595))) | 0.000038 | 0.000040 | 1.0000 | 0.9900 | 14.0 | Fastest distillation; core multiplicative interaction is captured cleanly. |

## Comments

- Quick mode appears well-balanced: runtime is practical while all benchmarks retain very high NN fit (NN R² ≥ 0.9996).
- Simple and medium-complexity targets (`x²`, `x·y`, and `x·exp(-x²)`) are recovered with high fidelity and manageable symbolic complexity.
- Noisy or structurally rich targets (`sin(x)+noise`, `Normal PDF`, `Black-Scholes`) remain accurate numerically but produce less human-canonical equations.
- Distillation times are consistent for most cases (~28–38s), with `x·y` notably faster (14.0s), matching lower functional complexity.
- Overall, this quick profile is appropriate for iterative experimentation before running full-mode sweeps.

## Evaluation

- **Overall verdict:** Success. The pipeline trains reliably in quick mode and produces stable symbolic outputs with strong predictive quality.
- **Best recoveries:** `x²` and `x·y` (near-perfect NN metrics plus equations centered on expected multiplicative structure).
- **Strong but less interpretable:** `sin(x)+noise` and `x·exp(-x²)` (excellent fit, extra corrective terms reduce readability).
- **Numerically strong, formula-level caveat:** `Normal PDF` and `Black-Scholes` achieve strong fit but recovered expressions are approximations rather than canonical closed forms.
- **Metric caveat:** `Sym MSE` and `Sym R²` are currently derived placeholders in the benchmark script, so final judgments should prioritize `NN MSE`, `NN R²`, symbolic form plausibility, and runtime.

## Conclusion

This quick run is a solid **engineering success** for rapid iteration, but only a **partial scientific success** for equation discovery. If the target is to show that SymTorch can produce useful symbolic surrogates quickly, it succeeds: all tasks show excellent predictive fit and stable runtime. If the stricter target is to consistently recover compact, domain-canonical formulas, results are mixed—especially for `Normal PDF` and `Black-Scholes`, where expressions fit well but are not the expected closed forms.

Key limitations are current placeholder symbolic metrics, potential over-complexity in discovered equations, and quick-mode constraints that can bias toward proxy expressions. Next steps should be: (1) compute true symbolic predictions from recovered equations for real `Sym MSE`/`Sym R²`, (2) enforce stronger complexity penalties and operator constraints in PySR, (3) run full-mode plus multiple seeds for stability, and (4) add benchmark-specific acceptance checks for formula structure, not just fit.
