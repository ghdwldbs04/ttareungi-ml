
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

features = pd.read_csv(
    PROCESSED / "subway_features_new.csv",
    dtype={"station": str},
    low_memory=False
)

sources = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "도시철도역사정보" in nfc(p.name)
]

if len(sources) != 1:
    raise ValueError("지하철 원본 파일 확인 필요")

raw = pd.read_excel(sources[0])

print("대여소:", len(stations))
print("새 피처:", len(features))
print("원본 역사:", len(raw))

print("\n" + "=" * 70)
print("2. 대여소와 새 피처 병합")

merged = stations[
    ["city", "station", "lat", "lon"]
].merge(
    features,
    on=["city", "station"],
    how="left",
    validate="one_to_one",
    indicator=True
)

if not (merged["_merge"] == "both").all():
    raise ValueError("병합 누락")

if merged["dist_subway_m"].isna().any():
    raise ValueError("새 거리 결측")

print("병합 성공:", len(merged))

print("\n" + "=" * 70)
print("3. 원본 전체 역사 기준 최근접 거리 계산")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

sx, sy = transformer.transform(
    merged["lon"].to_numpy(),
    merged["lat"].to_numpy()
)

rx, ry = transformer.transform(
    raw["역경도"].to_numpy(),
    raw["역위도"].to_numpy()
)

station_xy = np.column_stack([sx, sy])
raw_xy = np.column_stack([rx, ry])

tree = cKDTree(raw_xy)

raw_dist, raw_idx = tree.query(
    station_xy,
    k=1
)

merged["raw_dist"] = raw_dist

merged["raw_station_name"] = (
    raw.iloc[raw_idx]["역사명"]
    .to_numpy()
)

merged["raw_line"] = (
    raw.iloc[raw_idx]["노선명"]
    .to_numpy()
)

merged["diff_m"] = (
    merged["dist_subway_m"]
    - merged["raw_dist"]
)

merged["abs_diff_m"] = merged["diff_m"].abs()

different = merged[
    merged["abs_diff_m"] > 1
].copy()

print("\n" + "=" * 70)
print("4. 도시별 불일치 현황")

print(
    different["city"]
    .value_counts()
    .to_string()
)

print("\n전체 불일치:", len(different))

print("\n" + "=" * 70)
print("5. 거리 차이 방향 확인")

print(
    "새 거리가 더 긴 경우:",
    int((different["diff_m"] > 0).sum())
)

print(
    "새 거리가 더 짧은 경우:",
    int((different["diff_m"] < 0).sum())
)

print("\n차이 분포(m):")

if len(different):
    print(
        different["abs_diff_m"]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
        .round(3)
        .to_string()
    )

print("\n" + "=" * 70)
print("6. 거리 차이 상위 30개")

print(
    different.sort_values(
        "abs_diff_m",
        ascending=False
    )
    .head(30)[
        [
            "city",
            "station",
            "raw_station_name",
            "raw_line",
            "raw_dist",
            "dist_subway_m",
            "diff_m"
        ]
    ]
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("7. 원본 최근접 역별 불일치 빈도")

if len(different):
    print(
        different.groupby(
            ["raw_station_name", "raw_line"]
        )
        .size()
        .sort_values(ascending=False)
        .head(25)
        .to_string()
    )

print("\n" + "=" * 70)
print("8. 기본 논리 검사")

# 새 피처가 원본 좌표의 일부만 남겨서
# 계산된 것이라면 새 거리는 원본 거리보다
# 짧아질 수 없음
unexpected = merged[
    merged["diff_m"] < -1
]

print(
    "새 거리가 원본보다 1m 이상 짧은 경우:",
    len(unexpected)
)

if len(unexpected):
    print(
        unexpected[
            [
                "city",
                "station",
                "raw_dist",
                "dist_subway_m",
                "diff_m"
            ]
        ]
        .head(20)
        .round(3)
        .to_string(index=False)
    )

print("\n지하철 좌표 통합 영향 조사 완료!")
print("파일 저장 및 수정 없음")
