"""GPU smoke test: train the toy surrogate on CUDA and CPU and compare time / accuracy.

    .\\.venv\\Scripts\\python.exe scripts/gpu_smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heatgraph.surrogate.graphs import SceneGraph  # noqa: E402
from heatgraph.surrogate.model import HeatGraphGNN  # noqa: E402
from heatgraph.surrogate.train import evaluate, fit  # noqa: E402
from heatgraph.synthetic import generate_labels, random_plantings, toy_scene  # noqa: E402


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    s = toy_scene()
    X = random_plantings(s, 60, 0.35, 1)
    Y = generate_labels(s, X)
    sg = SceneGraph(s, baseline_utci=Y[0])
    tr, va, te = sg.dataset(X[:40], Y[:40]), sg.dataset(X[40:50], Y[40:50]), sg.dataset(X[50:], Y[50:])
    for dev in ("cuda", "cpu"):
        torch.manual_seed(0)
        m = HeatGraphGNN(hidden=48)
        t = time.time()
        fit(m, tr, va, epochs=60, device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t
        r = evaluate(m, te, device=dev)
        on = {p.device.type for p in m.parameters()}
        left = {d[k].x.device.type for d in tr for k in ("segment", "slot")}
        assert left == {"cpu"}, f"training data was moved off the CPU: {left}"
        print(f"{dev:5s} train {dt:5.1f} s | test MAE {r['mae']:.3f} C | params on {on} | data still on {left}")
    print("peak CUDA memory MB:", round(torch.cuda.max_memory_allocated() / 2**20, 1))


if __name__ == "__main__":
    main()
