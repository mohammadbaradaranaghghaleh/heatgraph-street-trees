"""Thin wrapper so the ingestion runs without installing the package:

    py -3 scripts/01_ingest.py                 # dry-run (default)
    py -3 scripts/01_ingest.py --execute --patch ams_grachtengordel --layer weather
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heatgraph.ingest.run import main  # noqa: E402

raise SystemExit(main())
