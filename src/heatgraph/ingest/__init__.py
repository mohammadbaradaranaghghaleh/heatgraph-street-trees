"""Block A: data ingestion. Every fetcher has the same contract:

    plan(patch, ctx) -> list[Request]      # stdlib only, no network (dry-run)
    fetch(patch, ctx) -> list[Path]        # performs I/O, writes under data/raw/<patch_id>/

`run.py` defaults to dry-run; network access needs an explicit --execute.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    source: str
    url: str
    est_mb: float
    note: str = ""
