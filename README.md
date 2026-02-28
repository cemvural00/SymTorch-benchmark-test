# SymTorch Benchmark (Quick + Full)

A lightweight benchmark suite to test symbolic distillation with SymTorch + PySR on:
- `x²`
- `sin(x)` (noisy)
- `x·exp(-x²)`
- Normal PDF
- Black-Scholes call pricing (2D input)
- `x·y` (2D interaction)

## What this project does
- Trains small MLPs on known target functions.
- Distills trained models into symbolic equations with SymTorch/PySR.
- Reports both neural-network and symbolic metrics.
- Writes a Markdown report for each run.

## Requirements
- Python 3.11 recommended
- Julia (PySR may install/configure it automatically on first run)

Install packages:

```bash
pip install -r requirements.txt
```

## Run
Quick mode (faster iteration):

```bash
python symtorch_benchmark.py --quick --report quick_benchmark_report.md
```

Full mode:

```bash
python symtorch_benchmark.py --report benchmark_report.md
```

If `--report` is omitted, an auto-named report is generated in the project root.

## Output
- Markdown summary report in project root (e.g. `quick_benchmark_report.md`)
- PySR artifacts in `SR_output/`

## Notes
- Quick mode reduces samples/epochs/iterations, while keeping learning meaningful.
- Training uses `ReduceLROnPlateau` and early stopping.
- Symbolic metrics are computed from symbolic model predictions (not placeholders).

## Main files
- `symtorch_benchmark.py` — benchmark runner
- `requirements.txt` — Python dependencies
- `symtorch_benchmark_guide.md` — benchmark design notes
- `quick_benchmark_report_v2.md` — latest example report
