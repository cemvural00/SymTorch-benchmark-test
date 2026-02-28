# SymTorch Benchmark Report

- Generated: 2026-02-28T16:37:54
- Mode: quick
- Total runtime (s): 275.0

## Summary

| Benchmark | Ground truth | Recovered equation | NN MSE | Sym MSE | NN R² | Sym R² | Distill time (s) | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| x² | x² | (((((exp(sin(exp(x)) + x) * -0.00010363123) + x) + 0.0010386093) * x) + 0.0004612267) * 1.0000564 | 0.000005 | 0.000003 | 1.0000 | 1.0000 | 24.4 | Very strong recovery; dominant quadratic structure with very small corrective terms. |
| sin(x)+noise | sin(x) | (0.007791113 * sin(-0.4386063 + (inv(x) + (x * 1.4854046)))) + sin(x + 0.004858796) | 0.000149 | 0.000020 | 0.9997 | 1.0000 | 10.9 | Robust under noise; main `sin(x)` term is recovered clearly, plus a small residual term. |
| x·exp(-x²) | x * exp(-x²) | (x + -0.0034902166) * ((exp((x * -1.0123333) * (x + -0.01155417)) + 0.0010968383) * 1.0078841) | 0.000022 | 0.000007 | 0.9996 | 0.9999 | 11.2 | High-fidelity structural match to target; minor offsets and scale tweaks remain. |
| Normal PDF | exp(-x²/2) / sqrt(2π)  ≈  0.3989 * exp(-0.5·x²) | (inv(exp((x * 0.4981684) * x)) * 0.39874282) + (sin((x * 2.844761) + -1.3982816) * -0.0009131053) | 0.000001 | 0.000000 | 1.0000 | 1.0000 | 13.2 | Near-canonical Gaussian core discovered (`0.3987 * exp(-0.498x²)`), with negligible oscillatory correction. |
| Black-Scholes | S·Φ(d1) - K·e^{-rT}·Φ(d2)  [closed form] | (((sigma * (((T + -0.51682204) * (inv(T) * 3.077431)) + 33.20615)) + sigma) + (((sigma * ((T * -13.659476) + 65.458565)) + (T + 5.890318)) * T)) * 0.42902872 | 0.006797 | 0.028044 | 0.9999 | 0.9997 | 12.7 | Good surrogate fit in sampled regime, but formula is not interpretable as canonical Black-Scholes structure. |
| x·y | x * y | x * (((sin((sin((x + sin(x)) + 1.1828345) + -0.19968086) + (y + (y * x))) * -0.0038944874) + y) * 0.99933445) | 0.000038 | 0.000000 | 1.0000 | 1.0000 | 6.6 | Fastest case; interaction `x*y` is recovered cleanly with tiny residual modulation. |

## Comments

- This v2 quick run is materially better than the previous quick run on symbolic fidelity: most tasks now show symbolic metrics at or near NN quality.
- Runtime is excellent for iterative loops (total 275.0s, with individual distillation mostly ~7–13s except `x²` at 24.4s).
- For analytic functions (`x²`, `x·exp(-x²)`, `Normal PDF`, `x·y`), recovered expressions are not only accurate but also structurally close to expected forms.
- The noisy trigonometric case remains stable and interpretable, with a dominant `sin(x)` term plus small correction components.
- `Black-Scholes` remains the key realism gap: predictive fit is high, but expression complexity and missing canonical terms suggest approximation rather than true formula discovery.

## Evaluation

- **Target check (quick mode):** Achieved for practical benchmarking. The pipeline now provides fast runs with high NN and symbolic predictive quality on all six tasks.
- **Interpretability check:** Partially achieved. Four tasks are close to expected structure; `Black-Scholes` still fails strict formula interpretability.
- **Generalization confidence:** Moderate-to-high for smooth analytic tasks, moderate for finance-style structured formulas where multiple proxies can fit sampled domains.
- **Metric quality:** Improved and trustworthy compared to earlier placeholder-based reporting; `Sym MSE`/`Sym R²` now reflect actual symbolic predictions.
- **Critical read:** The jump from high fit to true mechanism discovery is still the main challenge; high R² alone should not be treated as symbolic correctness.

## Conclusion

This v2 report is a strong step forward: as a **rapid symbolic surrogate workflow**, it is performing well and meeting its intended quick-iteration target. The system now demonstrates both speed and consistency, with several recovered equations that are close to textbook forms. That said, as a **strict scientific formula-recovery benchmark**, it is not fully there yet because `Black-Scholes` remains approximation-heavy and structurally non-canonical.

The next practical upgrades are: (1) add out-of-domain validation splits to test whether symbolic forms hold outside training ranges, (2) tighten PySR operator/complexity constraints for finance benchmarks, (3) run multi-seed quick and full modes to quantify stability, and (4) add benchmark-specific structure checks (for example, enforcing Gaussian or Black-Scholes motifs where appropriate). With those additions, this can move from "high-quality approximation" toward "reliable formula discovery." 
