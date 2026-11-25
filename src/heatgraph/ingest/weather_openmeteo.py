"""Hourly meteorological forcing from the Open-Meteo historical archive (ERA5-based, no key).

Stdlib only. Wind is requested in m/s at 10 m; Block C reduces it to 1.1 m.
For the paper, ERA5 itself (Hersbach et al., 2020) is the cited source; Open-Meteo is the
access layer and is named in the data-availability statement.
"""
from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

from . import Request

API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation",
          "direct_normal_irradiance", "diffuse_radiation", "cloud_cover", "surface_pressure"]


def _url(lat: float, lon: float, day: str) -> str:
    # include the previous UTC day so the LOCAL day (UTC+1/+2 in NL/DE) is complete from 00:00
    from datetime import date, timedelta

    start = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": start, "end_date": day,
         "hourly": ",".join(HOURLY), "wind_speed_unit": "ms", "timezone": "UTC"}
    return f"{API}?{urllib.parse.urlencode(q)}"


def plan(patch, ctx) -> list[Request]:
    return [Request("open_meteo", _url(patch.lat, patch.lon, d), est_mb=0.01) for d in ctx.heatwave_days]


def fetch(patch, ctx) -> list[Path]:
    out = ctx.patch_dir(patch)
    path = out / "weather_hourly.csv"
    rows = []
    for day in ctx.heatwave_days:
        req = urllib.request.Request(_url(patch.lat, patch.lon, day),
                                     headers={"User-Agent": "HeatGraph-research/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        h = data["hourly"]
        for i, t in enumerate(h["time"]):
            rows.append({"time_utc": t, **{k: h[k][i] for k in HOURLY}})
    rows = list({r["time_utc"]: r for r in rows}.values())      # de-duplicate overlapping days
    rows.sort(key=lambda r: r["time_utc"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time_utc", *HOURLY])
        w.writeheader()
        w.writerows(rows)
    return [path]
