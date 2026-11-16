"""Sample filters for the robustness analysis (Section 5.2). Each returns ``dataset_index`` values."""
from __future__ import annotations

import pandas as pd

# Approximate latitude/longitude boxes of the provinces and territories used in Table 8.
PROVINCE_BOXES = {
    "YT": dict(lat=(60.0, 70.0), lon=(-180.0, -124.0)),
    "NWT": dict(lat=(60.0, 78.0), lon=(-124.0, -102.0)),
    "BC": dict(lat=(49.0, 60.0), lon=(-180.0, -120.0)),
    "AB": dict(lat=(49.0, 60.0), lon=(-120.0, -110.0)),
    "SK": dict(lat=(49.0, 60.0), lon=(-110.0, -102.0)),
    "MB": dict(lat=(49.0, 60.0), lon=(-102.0, -90.0)),
    "ON": dict(lat=(41.5, 57.0), lon=(-95.0, -74.0)),
    "QC": dict(lat=(45.0, 63.0), lon=(-79.5, -57.0)),
}


def _indices(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    return df.loc[mask, "dataset_index"].tolist()


def by_size(df: pd.DataFrame, min_ha: float = 0.0, max_ha: float = float("inf")) -> list[int]:
    """Samples whose largest fire in the tile is in ``[min_ha, max_ha)`` hectares."""
    return _indices(df, (df["max_fire_size"] >= min_ha) & (df["max_fire_size"] < max_ha))


def by_fireday(df: pd.DataFrame, min_days: int = 1, max_days: int = 10**6) -> list[int]:
    """Samples whose fire age (days since the first detection) is in ``[min_days, max_days]``."""
    return _indices(df, (df["fireday"] >= min_days) & (df["fireday"] <= max_days))


def by_year(df: pd.DataFrame, year: int) -> list[int]:
    return _indices(df, df["year"] == year)


def by_month(df: pd.DataFrame, month: int) -> list[int]:
    return _indices(df, df["parsed_date"].dt.month == month)


def by_province(df: pd.DataFrame, province: str) -> list[int]:
    """Samples whose tile centre falls in the box of ``province`` (see :data:`PROVINCE_BOXES`).

    The lower longitude bound is exclusive except for the westernmost boxes.
    """
    box = PROVINCE_BOXES[province.upper()]
    (lat_min, lat_max), (lon_min, lon_max) = box["lat"], box["lon"]
    lat, lon = df["center_lat"], df["center_lon"]
    lat_ok = (lat >= lat_min) & ((lat <= lat_max) if province.upper() in ("YT", "NWT") else (lat < lat_max))
    lon_ok = ((lon > lon_min) | (lon_min == -180.0)) & (lon <= lon_max)
    return _indices(df, lat_ok & lon_ok)
