
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
STATIONS = BASE / "processed" / "stations_clean.csv"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. 경사도 파일 및 대여소 읽기")

slope_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and (
        "Slope" in nfc(p.name)
        or "경사" in nfc(p.name)
    )
]

if len(slope_dirs) != 1:
    raise ValueError("경사도 폴더 확인 필요")

candidates = [
    p for p in slope_dirs[0].rglob("*.tif")
    if nfc(p.name).lower() == "slope_all.tif"
]

if len(candidates) != 1:
    raise ValueError("Slope_All.tif 확인 필요")

RASTER = candidates[0]

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

if len(stations) != 5224:
    raise ValueError("대여소 수 확인 필요")

if stations[["lat", "lon"]].isna().any().any():
    raise ValueError("대여소 좌표 결측")

if stations.duplicated(
    ["city", "station"]
).any():
    raise ValueError("대여소 키 중복")

print("대여소 수:", len(stations))
print("래스터:", nfc(RASTER.relative_to(BASE)))

print("\n" + "=" * 70)
print("2. 전체 대여소 경사도 추출")

with rasterio.open(RASTER) as src:
    if src.crs is None:
        raise ValueError("래스터 좌표계 없음")

    xs, ys = transform(
        "EPSG:4326",
        src.crs,
        stations["lon"].tolist(),
        stations["lat"].tolist()
    )

    coords = list(zip(xs, ys))

    inside = np.array(
        [
            (
                src.bounds.left <= x < src.bounds.right
                and src.bounds.bottom < y <= src.bounds.top
            )
            for x, y in coords
        ],
        dtype=bool
    )

    sampled = src.sample(
        coords,
        indexes=1,
        masked=True
    )

    values = []

    for item in sampled:
        value = item[0]

        if np.ma.is_masked(value):
            values.append(np.nan)
        else:
            values.append(float(value))

stations["slope_value"] = values
stations["inside_raster"] = inside

print("래스터 범위 안:", int(inside.sum()))
print("래스터 범위 밖:", int((~inside).sum()))

print(
    "경사도 결측:",
    int(stations["slope_value"].isna().sum())
)

print(
    "경사도 0:",
    int((stations["slope_value"] == 0).sum())
)

print(
    "경사도 음수:",
    int((stations["slope_value"] < 0).sum())
)

print(
    "경사도 무한대:",
    int(
        np.isinf(
            stations["slope_value"].to_numpy()
        ).sum()
    )
)

print("\n" + "=" * 70)
print("3. 도시별 경사도 분포")

summary = (
    stations.groupby("city")["slope_value"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max"
    )
)

summary["missing"] = (
    stations.groupby("city")["slope_value"]
    .apply(lambda x: x.isna().sum())
)

summary["zero_count"] = (
    stations.groupby("city")["slope_value"]
    .apply(lambda x: (x == 0).sum())
)

summary["zero_pct"] = (
    summary["zero_count"]
    / stations.groupby("city").size()
    * 100
)

print(summary.round(3).to_string())

print("\n" + "=" * 70)
print("4. 전체 경사도 분위수")

print(
    stations["slope_value"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    )
    .round(4)
    .to_string()
)

print("\n" + "=" * 70)
print("5. 경사도가 큰 대여소 상위 20개")

print(
    stations[
        [
            "city",
            "station",
            "lat",
            "lon",
            "slope_value"
        ]
    ]
    .sort_values(
        "slope_value",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("6. 같은 좌표의 대여소 검사")

coord_groups = (
    stations.groupby(["lat", "lon"])
    .agg(
        station_count=("station", "size"),
        slope_unique=(
            "slope_value",
            lambda x: x.nunique(dropna=False)
        )
    )
)

duplicate_coords = coord_groups[
    coord_groups["station_count"] > 1
]

inconsistent = duplicate_coords[
    duplicate_coords["slope_unique"] > 1
]

print(
    "대여소가 2개 이상인 좌표:",
    len(duplicate_coords)
)

print(
    "같은 좌표에서 경사도가 다른 경우:",
    len(inconsistent)
)

print("\n전체 대여소 경사도 검사 완료!")
