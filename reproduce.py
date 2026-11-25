"""Single-command reproduction of the HeatGraph paper.

    python reproduce.py                 # default: regenerate + verify all figures and tables (minutes, CPU ok)
    python reproduce.py --tests         # also run the full unit-test suite
    python reproduce.py --full --i-understand-this-overwrites-results   # whole pipeline from downloads (hours, GPU)

Default mode (non-destructive)
  1. regenerates the 7 figures and 6 tables from the released results into reproduction/
     (never touches figures/, tables/, manuscript/ or data/);
  2. compares every table byte-for-byte with tables/ and every figure pixel-wise with figures/*.png;
  3. rebuilds the manuscript's numeric facts in a scratch root and compares them with manuscript/facts.json;
  4. runs manuscript/check_numbers.py (text <-> facts audit).
  Exit code 0 only if everything matches.

Full mode re-runs ingestion, freezing, calibration against SOLWEIG-GPU (conda env `solweig-ref`),
dataset generation, training, benchmarks and verification, then the default mode. Upstream data
(3DBAG, Overture, canopy tiles, weather) can change between releases and GPU training is not
bit-deterministic, so full-mode numbers will be close to, not identical with, the paper. Run it in a
fresh clone; it refuses to overwrite existing results without the explicit flag.
"""
from __future__ import annotations

import argparse
import filecmp
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "reproduction"
FIG_TOL = 0.02          # max fraction of differing pixels per figure (F3 recomputes GNN predictions)
FACT_RTOL = 1e-6

REQUIRED = [  # released result files needed by the default mode
    "data/raw/ingest_summary.csv", "data/raw/FROZEN.json", "data/validation/calibration.json",
    "data/models/metrics_edgemlp.json", "data/models/metrics_gatv2.json",
    "data/models/metrics_edgemlp_v1_patchval.json", "data/models/metrics_edgemlp_v2_patchval.json",
    "data/models/metrics_linear_physics.json", "data/models/density_analysis.json",
    "data/models/physics_feature_corr.json", "data/models/heatgraph_edgemlp_v2_patchval.pt",
    "data/stage5/conformal.json", "data/stage5/rot_centrum/solweig_verification.json",
    "data/stage5/rot_centrum/solweig_verification_arrays.npz",
] + [f"data/stage5/{p}/{f}" for p in ("rot_centrum", "rot_kop_van_zuid", "rot_oude_westen", "xfer_freiburg_altstadt")
     for f in ("results.json", "designs.npz")] \
  + [f"data/processed/{p}/{f}" for p in ("rot_centrum", "rot_kop_van_zuid", "rot_oude_westen", "xfer_freiburg_altstadt")
     for f in ("scene.npz", "scene.json", "labels.npz", "weather.csv")]

FULL_PIPELINE = [
    ("ingest all patches (downloads)", ["scripts/01_ingest.py", "--execute"]),
    ("ingestion QA table", ["scripts/ingest_summary.py"]),
    ("freeze raw dataset (SHA-256)", ["scripts/freeze_dataset.py"]),
    ("engine calibration + SOLWEIG-GPU gate", ["scripts/02_solweig_benchmark.py", "--calibrate", "ams_grachtengordel:2019-07-25",
                                               "--evaluate", "utr_binnenstad:2022-07-19", "rot_oude_westen:2020-08-08",
                                               "ams_zuidas:2019-07-25", "rot_centrum:2022-07-19"]),
    ("counterfactual dataset (GPU)", ["scripts/04_generate_dataset.py"]),
    ("surrogates v1, within-patch validation (T3 rows 1-2)",
     ["scripts/05_train_surrogate.py", "--conv", "edgemlp", "--conv", "gatv2", "--features", "v1", "--val-mode", "planting",
      "--dropout", "0", "--weight-decay", "1e-5", "--patience", "1000", "--epochs", "60"]),
    ("surrogate v1, leave-patch-out", ["scripts/05_train_surrogate.py", "--conv", "edgemlp", "--features", "v1", "--epochs", "80"]),
    ("surrogate v2 physics edges", ["scripts/05_train_surrogate.py", "--conv", "edgemlp", "--features", "v2", "--epochs", "80"]),
    ("linear physics baseline", ["scripts/06_baselines.py"]),
    ("error by planting density", ["scripts/07_density_analysis.py"]),
    ("optimisation benchmark", ["scripts/08_benchmark.py"]),
    ("SOLWEIG-GPU verification", ["scripts/09_solweig_verify.py"]),
    ("physics-feature correlation", ["scripts/11_feature_correlation.py"]),
]
# file names expected by the figure script for the within-patch-validation runs
RENAMES = {"metrics_edgemlp_v1_plantingval.json": "metrics_edgemlp.json",
           "metrics_gatv2_v1_plantingval.json": "metrics_gatv2.json"}


def log(msg):
    print(f"[reproduce] {msg}", flush=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    spec.loader.exec_module(mod)
    return mod


def regenerate(out: Path):
    """Run scripts/10_figures_tables.py with its output folders redirected to `out`."""
    m = load_module(ROOT / "scripts" / "10_figures_tables.py", "figures_tables")
    m.FIG, m.TAB = out / "figures", out / "tables"
    m.FIG.mkdir(parents=True, exist_ok=True)
    m.TAB.mkdir(parents=True, exist_ok=True)
    m.main()


def compare_tables(out: Path) -> list[str]:
    bad = []
    for f in sorted((ROOT / "tables").glob("*")):
        g = out / "tables" / f.name
        if not g.exists() or not filecmp.cmp(f, g, shallow=False):
            bad.append(f"table {f.name} differs")
    return bad


def compare_figures(out: Path) -> tuple[list[str], dict]:
    import matplotlib.image as mpimg
    import numpy as np

    bad, report = [], {}
    for f in sorted((ROOT / "figures").glob("*.png")):
        g = out / "figures" / f.name
        if not g.exists():
            bad.append(f"figure {f.name} not regenerated")
            continue
        a, b = mpimg.imread(f), mpimg.imread(g)
        if a.shape != b.shape:
            bad.append(f"figure {f.name}: size {a.shape} vs {b.shape}")
            continue
        frac = float((np.abs(a - b).max(axis=-1) > 1 / 255).mean())
        report[f.name] = frac
        if frac > FIG_TOL:
            bad.append(f"figure {f.name}: {100 * frac:.2f}% pixels differ (> {100 * FIG_TOL:.0f}%)")
    return bad, report


def compare_facts() -> list[str]:
    """Rebuild manuscript facts in a scratch root (configs copied, data read in place)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "configs", tmp / "configs")
        (tmp / "manuscript").mkdir()
        m = load_module(ROOT / "manuscript" / "build_facts.py", "build_facts_repro")
        m.ROOT = tmp
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            m.main()
        new = json.loads((tmp / "manuscript" / "facts.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "manuscript" / "facts.json").read_text(encoding="utf-8"))
    bad = []

    def walk(a, b, path):
        if isinstance(a, dict):
            if set(a) != set(b):
                bad.append(f"facts {path}: keys differ")
            for k in a:
                if k in b:
                    walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            if len(a) != len(b):
                bad.append(f"facts {path}: length differs")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, float) or isinstance(b, float):
            if abs(float(a) - float(b)) > FACT_RTOL * max(1.0, abs(float(a))):
                bad.append(f"facts {path}: {a} vs {b}")
        elif a != b:
            bad.append(f"facts {path}: {a} vs {b}")

    walk(old, new, "facts")
    return bad


def run(cmd, cwd=ROOT):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([PY, "-W", "ignore", *cmd], cwd=cwd).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", action="store_true", help="also run the unit-test suite")
    ap.add_argument("--full", action="store_true", help="re-run the whole pipeline from downloads first")
    ap.add_argument("--i-understand-this-overwrites-results", dest="force", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()

    if a.full:
        if (ROOT / "data" / "stage5").exists() and not a.force:
            log("refusing --full: data/stage5 exists (released results). Use a fresh clone, or add "
                "--i-understand-this-overwrites-results.")
            return 2
        for desc, cmd in FULL_PIPELINE:
            if cmd[0].endswith("freeze_dataset.py") and (ROOT / "data" / "raw" / "FROZEN.json").exists():
                log("STEP skipped: keeping the paper's data/raw/FROZEN.json; dataset generation verifies the new "
                    "downloads against it (delete it to re-freeze changed upstream data deliberately)")
                continue
            log(f"STEP: {desc}")
            if run(cmd) != 0:
                log(f"FAILED at: {desc}")
                return 1
            if cmd[0].endswith("05_train_surrogate.py"):
                for src, dst in RENAMES.items():
                    if (ROOT / "data" / "models" / src).exists():
                        shutil.copyfile(ROOT / "data" / "models" / src, ROOT / "data" / "models" / dst)

    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        log("missing released result files (download the results archive, see README):")
        for p in missing:
            log(f"  - {p}")
        return 2

    out = OUT / time.strftime("%Y%m%d-%H%M%S")
    log(f"regenerating figures and tables into {out.relative_to(ROOT)}")
    regenerate(out)
    problems = compare_tables(out)
    fig_bad, fig_report = compare_figures(out)
    problems += fig_bad
    log("rebuilding manuscript facts in a scratch root")
    problems += compare_facts()
    log("auditing manuscript text against facts")
    if run(["check_numbers.py"], cwd=ROOT / "manuscript") != 0:
        problems.append("manuscript/check_numbers.py failed")
    if a.tests:
        log("running unit tests")
        if run(["-m", "unittest", "discover", "-s", "tests"]) != 0:
            problems.append("unit tests failed")

    summary = {"output": str(out.relative_to(ROOT)), "tables_identical": not any(p.startswith("table") for p in problems),
               "figure_pixel_diff_fraction": fig_report, "problems": problems,
               "seconds": round(time.time() - t0, 1)}
    (out / "reproduction_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"figures: max differing-pixel fraction {max(fig_report.values(), default=0):.4f} (tolerance {FIG_TOL})")
    if problems:
        log("MISMATCHES:")
        for p in problems:
            log(f"  - {p}")
        return 1
    log(f"OK: 6 tables byte-identical, 7 figures within tolerance, facts and manuscript consistent "
        f"({summary['seconds']:.0f} s). Report: {out.relative_to(ROOT) / 'reproduction_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
