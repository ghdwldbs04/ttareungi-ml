
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. 파일 읽기")

stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str},
    low_memory=False
)

subway_features = pd.read_csv(
    PROCESSED / "subway_features_new.csv",
    dtype={"station": str},
    low_memory=False
)

sources = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "도시철도역사정보" in nfc(p.name)
]

if len(sources) != 1:
    raise ValueError("지하철 원본 엑셀 확인 필요")

raw = pd.read_excel(sources[0])

print("대여소:", len(stations))
print("새 지하철 피처:", len(subway_features))
print("원본 역사 기록:", len(raw))

required = [
    "역사명", "노선명", "역위도", "역경도"
]

if not all(c in raw.columns for c in required):
    raise ValueError("원본 역사 열 확인 필요")

if raw[["역위도", "역경도"]].isna().any().any():
    raise ValueError("역 좌표 결측")

print("\n" + "=" * 70)
print("2. 기존 거리와 새 거리 비교")

old = stations[
    ["city", "station", "lat", "lon", "dist_subway_m"]
].rename(
    columns={"dist_subway_m": "old_dist"}
)

new = subway_features[
    ["city", "station", "dist_subway_m"]
].rename(
    columns={"dist_subway_m": "new_dist"}
)

merged = old.merge(
    new,
    on=["city", "station"],
    how="left",
    validate="one_to_one"
)

if merged["new_dist"].isna().any():
    raise ValueError("새 거리 결측")

merged["new_capped"] = merged["new_dist"].clip(
    upper=5000
)

merged["abs_diff"] = (
    merged["old_dist"] - merged["new_capped"]
).abs()

seoul = merged[
    merged["city"] == "seoul"
].copy()

top25 = (
    seoul.sort_values(
        "abs_diff",
        ascending=False
    )
    .head(25)
    .copy()
)

print("서울 대여소:", len(seoul))
print("차이 상위 대여소:", len(top25))

print("\n" + "=" * 70)
print("3. 원본 역사 전체 기준 최근접 역 계산")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

station_x, station_y = transformer.transform(
    top25["lon"].to_numpy(),
    top25["lat"].to_numpy()
)

subway_x, subway_y = transformer.transform(
    raw["역경도"].to_numpy(),
    raw["역위도"].to_numpy()
)

tree = cKDTree(
    np.column_stack([subway_x, subway_y])
)

nearest_dist, nearest_idx = tree.query(
    np.column_stack([station_x, station_y]),
    k=1
)

nearest_raw = raw.iloc[
    nearest_idx
].reset_index(drop=True)

result = top25.reset_index(drop=True)[
    [
        "station",
        "old_dist",
        "new_dist",
        "abs_diff"
    ]
].copy()

result["nearest_station"] = (
    nearest_raw["역사명"].astype(str)
)

result["nearest_line"] = (
    nearest_raw["노선명"].astype(str)
)

result["nearest_raw_dist"] = nearest_dist

result["same_name_records"] = [
    int((raw["역사명"] == name).sum())
    for name in result["nearest_station"]
]

result["raw_vs_new_diff"] = (
    result["nearest_raw_dist"]
    - result["new_dist"]
).abs()

print(
    result.round(3).to_string(index=False)
)

print("\n" + "=" * 70)
print("4. 원본 전체 역 거리와 새 거리 일치 여부")

print(
    "1m 이내 일치:",
    int((result["raw_vs_new_diff"] <= 1).sum()),
    "/",
    len(result)
)

print(
    "1m 초과 차이:",
    int((result["raw_vs_new_diff"] > 1).sum())
)

print("\n" + "=" * 70)
print("5. 최근접 역 이름별 빈도")

print(
    result["nearest_station"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 70)
print("6. 같은 이름의 역 좌표 상세")

names = result.loc[
    result["same_name_records"] > 1,
    "nearest_station"
].unique()

for name in names:
    matches = raw[
        raw["역사명"] == name
    ][
        [
            "역사명",
            "노선명",
            "역위도",
            "역경도"
        ]
    ]

    print("\n역명:", name)
    print(matches.to_string(index=False))

print("\n최근접 지하철역 조사 완료!")
print("파일 저장 및 수정 없음")
