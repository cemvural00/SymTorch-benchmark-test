"""
SymTorch Benchmark Suite
========================
Tests symbolic distillation (pip install symtorch) across:
  1. Simple analytic functions (clean + noisy)
  2. Probability functions (Normal PDF, Normal CDF approximation)
  3. Black-Scholes option pricing
  4. Multi-input functions

Install deps first:
    pip install symtorch pysr torch numpy scipy matplotlib

Run:
    python symtorch_benchmark.py
    python symtorch_benchmark.py --quick
    python symtorch_benchmark.py --report my_results.md

Each benchmark trains a small MLP on the target function, then asks SymTorch
to recover a symbolic equation from the network's learned behavior.
We report:
  - Ground truth formula
  - Recovered symbolic formula (from PySR's Pareto front)
  - MSE of symbolic approximation vs. ground truth
  - MSE of the NN vs. ground truth (baseline)
"""

import math
import time
import warnings
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")

# ─── SymTorch import ───────────────────────────────────────────────────────────
try:
    from symtorch import SymbolicModel
    SYMTORCH_AVAILABLE = True
except ImportError:
    print("[WARN] symtorch not installed. Run: pip install symtorch")
    SYMTORCH_AVAILABLE = False

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_mlp(in_features: int, hidden: int = 64, depth: int = 3) -> nn.Sequential:
    """Simple MLP with tanh activations — good for smooth function approximation."""
    layers = [nn.Linear(in_features, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 2000, lr: float = 1e-3, batch_size: int = 256,
          verbose: bool = False,
          use_scheduler: bool = True,
          lr_factor: float = 0.5,
          lr_patience: int = 150,
          min_lr: float = 1e-6,
          early_stopping_patience: int = 400,
          early_stopping_min_delta: float = 1e-6) -> list:
    """Train model with Adam + MSELoss, return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )

    loss_fn = nn.MSELoss()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    best_loss = float("inf")
    best_state_dict = None
    stale_epochs = 0

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = loss_fn(model(X), y).item()

        losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = val_loss < (best_loss - early_stopping_min_delta)
        if improved:
            best_loss = val_loss
            stale_epochs = 0
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale_epochs += 1

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:4d} | loss {val_loss:.6f} | lr {current_lr:.2e}")

        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            if verbose:
                print(f"  early stopping at epoch {epoch:4d} | best loss {best_loss:.6f}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return losses


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def distill_model(model: nn.Module, X_np: np.ndarray,
                  var_names: list, niterations: int = 40):
    """
    Run SymTorch model-agnostic distillation.
    Returns (best symbolic equation string, symbolic predictions if available).
    """
    if not SYMTORCH_AVAILABLE:
        return "SymTorch not installed", None

    # Model-agnostic mode: wrap any callable
    def model_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return model(t).numpy().flatten()

    sym_model = SymbolicModel(model_fn)
    sym_model.distill(
        X_np,
        fit_params={"variable_names": var_names},
        sr_params={"niterations": niterations, "verbosity": 0},
    )

    # Return best equation from Pareto front.
    # Newer symtorch versions store regressors in `pysr_regressor` (often a dict
    # by output dimension), while older variants may expose `equations_` directly.
    all_eqs = []
    symbolic_pred = None

    direct_eqs = getattr(sym_model, "equations_", None)
    if direct_eqs is not None and hasattr(direct_eqs, "empty") and not direct_eqs.empty:
        all_eqs.append(direct_eqs)

    regressors = getattr(sym_model, "pysr_regressor", None)
    if isinstance(regressors, dict):
        candidates = regressors.values()
    elif regressors is not None:
        candidates = [regressors]
    else:
        candidates = []

    for reg in candidates:
        eqs = getattr(reg, "equations_", None)
        if eqs is not None and hasattr(eqs, "empty") and not eqs.empty:
            all_eqs.append(eqs)
        if symbolic_pred is None and hasattr(reg, "predict"):
            try:
                pred = reg.predict(X_np)
                symbolic_pred = np.asarray(pred).reshape(-1)
            except Exception:
                pass

    if all_eqs:
        try:
            import pandas as pd
            merged = pd.concat(all_eqs, ignore_index=True)
        except Exception:
            merged = all_eqs[0]

        if "loss" in merged.columns and "equation" in merged.columns:
            best = merged.sort_values("loss").iloc[0]
            return str(best["equation"]), symbolic_pred

    return "No equation found", symbolic_pred


def print_result(name, truth, nn_mse, sym_eq, sym_mse, nn_r2, sym_r2, elapsed):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  BENCHMARK: {name}")
    print(sep)
    print(f"  Ground truth    : {truth}")
    print(f"  Recovered eq    : {sym_eq}")
    print(f"  NN   MSE / R²   : {nn_mse:.6f}  /  {nn_r2:.4f}")
    print(f"  Sym  MSE / R²   : {sym_mse:.6f}  /  {sym_r2:.4f}")
    print(f"  Distill time    : {elapsed:.1f}s")
    print(sep)


def build_result(name, truth, nn_mse, sym_eq, sym_mse, nn_r2, sym_r2, elapsed):
    return {
        "name": name,
        "truth": truth,
        "sym_eq": sym_eq,
        "nn_mse": float(nn_mse),
        "sym_mse": float(sym_mse),
        "nn_r2": float(nn_r2),
        "sym_r2": float(sym_r2),
        "elapsed": float(elapsed),
    }


def scale_int(value: int, factor: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * factor)))


def write_markdown_report(results: list, report_path: Path, quick_mode: bool, total_elapsed: float):
    lines = [
        "# SymTorch Benchmark Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Mode: {'quick' if quick_mode else 'full'}",
        f"- Total runtime (s): {total_elapsed:.1f}",
        "",
        "## Summary",
        "",
        "| Benchmark | Ground truth | Recovered equation | NN MSE | Sym MSE | NN R² | Sym R² | Distill time (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for res in results:
        eq = str(res["sym_eq"]).replace("|", "\\|").replace("\n", " ")
        truth = str(res["truth"]).replace("|", "\\|")
        lines.append(
            f"| {res['name']} | {truth} | {eq} | {res['nn_mse']:.6f} | {res['sym_mse']:.6f} | {res['nn_r2']:.4f} | {res['sym_r2']:.4f} | {res['elapsed']:.1f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1 — x²  (clean, no noise, 1D)
# ═══════════════════════════════════════════════════════════════════════════════
def bench_quadratic(sample_factor: float = 1.0, epoch_factor: float = 1.0, iter_factor: float = 1.0):
    print("\n[1/6] x²  — simple quadratic (no noise)")
    N, noise = scale_int(2000, sample_factor, minimum=1000), 0.0
    x = np.random.uniform(-3, 3, (N, 1)).astype(np.float32)
    y_true = x ** 2 + noise * np.random.randn(N, 1).astype(np.float32)

    X = torch.tensor(x); Y = torch.tensor(y_true)
    model = make_mlp(1)
    train(model, X, Y, epochs=scale_int(3000, epoch_factor, minimum=1200))

    with torch.no_grad():
        nn_pred = model(X).numpy().flatten()
    nn_m = mse(y_true.flatten(), nn_pred)
    nn_r = r2(y_true.flatten(), nn_pred)

    t0 = time.time()
    eq, sym_pred = distill_model(model, x, ["x"], niterations=scale_int(40, iter_factor, minimum=20))
    elapsed = time.time() - t0

    if sym_pred is not None and len(sym_pred) == len(y_true.flatten()):
        sym_mse = mse(y_true.flatten(), sym_pred)
        sym_r = r2(y_true.flatten(), sym_pred)
    else:
        sym_mse = float("nan")
        sym_r = float("nan")

    print_result("x²", "x²", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)
    return build_result("x²", "x²", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2 — sin(x) + 0.1·noise  (1D, noisy)
# ═══════════════════════════════════════════════════════════════════════════════
def bench_sin_noisy(sample_factor: float = 1.0, epoch_factor: float = 1.0, iter_factor: float = 1.0):
    print("\n[2/6] sin(x) + noise — 1D, noisy")
    N = scale_int(3000, sample_factor, minimum=1200)
    x = np.random.uniform(-2 * math.pi, 2 * math.pi, (N, 1)).astype(np.float32)
    y_true = np.sin(x).astype(np.float32)
    y_noisy = (y_true + 0.1 * np.random.randn(N, 1)).astype(np.float32)

    X = torch.tensor(x); Y = torch.tensor(y_noisy)
    model = make_mlp(1, hidden=128, depth=4)
    train(model, X, Y, epochs=scale_int(4000, epoch_factor, minimum=1600))

    with torch.no_grad():
        nn_pred = model(X).numpy().flatten()
    nn_m = mse(y_true.flatten(), nn_pred)
    nn_r = r2(y_true.flatten(), nn_pred)

    t0 = time.time()
    eq, sym_pred = distill_model(model, x, ["x"], niterations=scale_int(50, iter_factor, minimum=25))
    elapsed = time.time() - t0

    if sym_pred is not None and len(sym_pred) == len(y_true.flatten()):
        sym_mse = mse(y_true.flatten(), sym_pred)
        sym_r = r2(y_true.flatten(), sym_pred)
    else:
        sym_mse = float("nan")
        sym_r = float("nan")
    print_result("sin(x) + noise", "sin(x)", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)
    return build_result("sin(x)+noise", "sin(x)", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3 — x·exp(-x²)  (1D, more structure)
# ═══════════════════════════════════════════════════════════════════════════════
def bench_gaussian_derivative(sample_factor: float = 1.0, epoch_factor: float = 1.0, iter_factor: float = 1.0):
    print("\n[3/6] x·exp(-x²) — Gaussian derivative shape")
    N = scale_int(3000, sample_factor, minimum=1200)
    x = np.random.uniform(-3, 3, (N, 1)).astype(np.float32)
    y_true = (x * np.exp(-x ** 2)).astype(np.float32)
    y_noisy = (y_true + 0.05 * np.random.randn(N, 1)).astype(np.float32)

    X = torch.tensor(x); Y = torch.tensor(y_noisy)
    model = make_mlp(1, hidden=128, depth=4)
    train(model, X, Y, epochs=scale_int(4000, epoch_factor, minimum=1600))

    with torch.no_grad():
        nn_pred = model(X).numpy().flatten()
    nn_m = mse(y_true.flatten(), nn_pred)
    nn_r = r2(y_true.flatten(), nn_pred)

    t0 = time.time()
    eq, sym_pred = distill_model(model, x, ["x"], niterations=scale_int(60, iter_factor, minimum=30))
    elapsed = time.time() - t0

    if sym_pred is not None and len(sym_pred) == len(y_true.flatten()):
        sym_mse = mse(y_true.flatten(), sym_pred)
        sym_r = r2(y_true.flatten(), sym_pred)
    else:
        sym_mse = float("nan")
        sym_r = float("nan")
    print_result("x·exp(-x²)", "x * exp(-x²)", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)
    return build_result("x·exp(-x²)", "x * exp(-x²)", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4 — Normal PDF  φ(x) = exp(-x²/2) / √(2π)
# ═══════════════════════════════════════════════════════════════════════════════
def bench_normal_pdf(sample_factor: float = 1.0, epoch_factor: float = 1.0, iter_factor: float = 1.0):
    print("\n[4/6] Normal PDF  φ(x) = exp(-x²/2) / √(2π)")
    N = scale_int(4000, sample_factor, minimum=1600)
    x = np.random.uniform(-4, 4, (N, 1)).astype(np.float32)
    y_true = scipy_norm.pdf(x).astype(np.float32)
    y_noisy = (y_true + 0.002 * np.random.randn(N, 1)).astype(np.float32)

    X = torch.tensor(x); Y = torch.tensor(y_noisy)
    model = make_mlp(1, hidden=128, depth=4)
    train(model, X, Y, epochs=scale_int(5000, epoch_factor, minimum=2000))

    with torch.no_grad():
        nn_pred = model(X).numpy().flatten()
    nn_m = mse(y_true.flatten(), nn_pred)
    nn_r = r2(y_true.flatten(), nn_pred)

    t0 = time.time()
    eq, sym_pred = distill_model(model, x, ["x"], niterations=scale_int(60, iter_factor, minimum=30))
    elapsed = time.time() - t0

    if sym_pred is not None and len(sym_pred) == len(y_true.flatten()):
        sym_mse = mse(y_true.flatten(), sym_pred)
        sym_r = r2(y_true.flatten(), sym_pred)
    else:
        sym_mse = float("nan")
        sym_r = float("nan")
    print_result(
        "Normal PDF",
        "exp(-x²/2) / sqrt(2π)  ≈  0.3989 * exp(-0.5·x²)",
        nn_m, eq, sym_mse, nn_r, sym_r, elapsed
    )
    return build_result(
        "Normal PDF",
        "exp(-x²/2) / sqrt(2π)  ≈  0.3989 * exp(-0.5·x²)",
        nn_m,
        eq,
        sym_mse,
        nn_r,
        sym_r,
        elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 5 — Black-Scholes Call Price  (2D input: S/K, T·σ²)
#
# Simplified: let moneyness m = ln(S/K)/(σ√T), cost_of_carry = r·T
# We parametrize with 2 inputs: [d1, d2]  where d2 = d1 - σ√T
# Price = S·Φ(d1) - K·e^{-rT}·Φ(d2)
#
# For benchmark we fix S=100, K=100, r=0.05 and sweep σ ∈ [0.05,0.8], T ∈ [0.1,2]
# Inputs:  [σ, T]     Output: call price
# Ground truth: Black-Scholes formula
# ═══════════════════════════════════════════════════════════════════════════════
def black_scholes_call(sigma: np.ndarray, T: np.ndarray,
                       S=100.0, K=100.0, r=0.05) -> np.ndarray:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)


def bench_black_scholes(sample_factor: float = 1.0, epoch_factor: float = 1.0, iter_factor: float = 1.0):
    print("\n[5/6] Black-Scholes call price  (inputs: σ, T)  ATM (S=K=100, r=0.05)")
    N = scale_int(5000, sample_factor, minimum=2000)
    sigma = np.random.uniform(0.05, 0.8, N).astype(np.float32)
    T = np.random.uniform(0.1, 2.0, N).astype(np.float32)
    y_true = black_scholes_call(sigma, T).astype(np.float32)
    y_noisy = (y_true + 0.5 * np.random.randn(N)).astype(np.float32)  # $0.5 noise

    X_np = np.stack([sigma, T], axis=1)
    X = torch.tensor(X_np); Y = torch.tensor(y_noisy.reshape(-1, 1))
    model = make_mlp(2, hidden=128, depth=4)
    train(model, X, Y, epochs=scale_int(5000, epoch_factor, minimum=2200))

    with torch.no_grad():
        nn_pred = model(X).numpy().flatten()
    nn_m = mse(y_true, nn_pred)
    nn_r = r2(y_true, nn_pred)

    t0 = time.time()
    eq, sym_pred = distill_model(model, X_np, ["sigma", "T"], niterations=scale_int(60, iter_factor, minimum=30))
    elapsed = time.time() - t0

    if sym_pred is not None and len(sym_pred) == len(y_true):
        sym_mse = mse(y_true, sym_pred)
        sym_r = r2(y_true, sym_pred)
    else:
        sym_mse = float("nan")
        sym_r = float("nan")
    print_result(
        "Black-Scholes Call",
        "S·Φ(d1) - K·e^{-rT}·Φ(d2)  [closed form]",
        nn_m, eq, sym_mse, nn_r, sym_r, elapsed
    )
    return build_result(
        "Black-Scholes",
        "S·Φ(d1) - K·e^{-rT}·Φ(d2)  [closed form]",
        nn_m,
        eq,
        sym_mse,
        nn_r,
        sym_r,
        elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 6 — x·y  (2D multiplication — tests multi-input recovery)
# ═══════════════════════════════════════════════════════════════════════════════
def bench_product(sample_factor: float = 1.0, epoch_factor: float = 1.0, iter_factor: float = 1.0):
    print("\n[6/6] x·y — 2D product (multi-input interaction test)")
    N = scale_int(3000, sample_factor, minimum=1200)
    x = np.random.uniform(-2, 2, (N, 1)).astype(np.float32)
    y = np.random.uniform(-2, 2, (N, 1)).astype(np.float32)
    z_true = (x * y).astype(np.float32)
    z_noisy = (z_true + 0.05 * np.random.randn(N, 1)).astype(np.float32)

    X_np = np.concatenate([x, y], axis=1)
    X = torch.tensor(X_np); Z = torch.tensor(z_noisy)
    model = make_mlp(2, hidden=64, depth=3)
    train(model, X, Z, epochs=scale_int(3000, epoch_factor, minimum=1200))

    with torch.no_grad():
        nn_pred = model(X).numpy().flatten()
    nn_m = mse(z_true.flatten(), nn_pred)
    nn_r = r2(z_true.flatten(), nn_pred)

    t0 = time.time()
    eq, sym_pred = distill_model(model, X_np, ["x", "y"], niterations=scale_int(40, iter_factor, minimum=20))
    elapsed = time.time() - t0

    if sym_pred is not None and len(sym_pred) == len(z_true.flatten()):
        sym_mse = mse(z_true.flatten(), sym_pred)
        sym_r = r2(z_true.flatten(), sym_pred)
    else:
        sym_mse = float("nan")
        sym_r = float("nan")
    print_result("x·y", "x * y", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)
    return build_result("x·y", "x * y", nn_m, eq, sym_mse, nn_r, sym_r, elapsed)


def parse_args():
    parser = argparse.ArgumentParser(description="SymTorch symbolic distillation benchmark suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster benchmark profile (fewer samples, epochs, and SR iterations).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to write markdown report. Default: auto-generated in project root.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    print("=" * 60)
    print("  SymTorch Symbolic Distillation Benchmark Suite")
    print("  pip install symtorch  |  Uses PySR under the hood")
    print("=" * 60)

    mode_label = "quick" if args.quick else "full"
    print(f"\nMode: {mode_label}")

    sample_factor = 0.65 if args.quick else 1.0
    epoch_factor = 0.45 if args.quick else 1.0
    iter_factor = 0.50 if args.quick else 1.0

    if args.report:
        report_path = Path(args.report)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"symtorch_benchmark_report_{mode_label}_{stamp}.md")

    torch.manual_seed(42)
    np.random.seed(42)

    if not SYMTORCH_AVAILABLE:
        print("\n[ERROR] Please install: pip install symtorch pysr")
        print("Then re-run this script.")
        return

    suite_start = time.time()
    results = []

    results.append(bench_quadratic(sample_factor, epoch_factor, iter_factor))
    results.append(bench_sin_noisy(sample_factor, epoch_factor, iter_factor))
    results.append(bench_gaussian_derivative(sample_factor, epoch_factor, iter_factor))
    results.append(bench_normal_pdf(sample_factor, epoch_factor, iter_factor))
    results.append(bench_black_scholes(sample_factor, epoch_factor, iter_factor))
    results.append(bench_product(sample_factor, epoch_factor, iter_factor))

    total_elapsed = time.time() - suite_start

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SUMMARY — Recovered Equations")
    print("═" * 60)
    for res in results:
        print(f"  {res['name']:<20s} →  {res['sym_eq']}")
    print("═" * 60)

    write_markdown_report(results, report_path, quick_mode=args.quick, total_elapsed=total_elapsed)

    print(f"\nReport saved to: {report_path}")
    print("Done. Check SR_output/ directory for full PySR equation tables.")


if __name__ == "__main__":
    main()
