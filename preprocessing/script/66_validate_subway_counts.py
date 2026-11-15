
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
print("1. 데이터 읽기")

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
    raise ValueError("지하철 원본 엑셀 확인 필요")

raw = pd.read_excel(sources[0])

print("대여소:", len(stations))
print("새 지하철 피처:", len(features))
print("원본 역사:", len(raw))

required = [
    "city",
    "station",
    "dist_subway_m",
    "subway_500",
    "subway_1000",
]

if not all(c in features.columns for c in required):
    raise ValueError("새 지하철 피처 열 확인 필요")

if features.duplicated(["city", "station"]).any():
    raise ValueError("새 지하철 피처 키 중복")

print("\n" + "=" * 70)
print("2. 대여소와 피처 병합")

merged = stations[
    ["city", "station", "lat", "lon", "subway_1000"]
].rename(
    columns={"subway_1000": "old_subway_1000"}
).merge(
    features,
    on=["city", "station"],
    how="left",
    validate="one_to_one",
    indicator=True
)

if not (merged["_merge"] == "both").all():
    raise ValueError("대여소 병합 누락")

print("병합:", len(merged), "/", len(stations))

print("\n" + "=" * 70)
print("3. 원본 역사 좌표 변환")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

station_x, station_y = transformer.transform(
    merged["lon"].to_numpy(),
    merged["lat"].to_numpy()
)

raw_x, raw_y = transformer.transform(
    raw["역경도"].to_numpy(),
    raw["역위도"].to_numpy()
)

station_xy = np.column_stack([station_x, station_y])
raw_xy = np.column_stack([raw_x, raw_y])

tree = cKDTree(raw_xy)

print("\n" + "=" * 70)
print("4. 원본 전체 역사 기준 거리 및 개수 계산")

raw_nearest, _ = tree.query(
    station_xy,
    k=1
)

raw_500 = np.array(
    [
        len(indices)
        for indices in tree.query_ball_point(
            station_xy,
            r=500
        )
    ]
)

raw_1000 = np.array(
    [
        len(indices)
        for indices in tree.query_ball_point(
            station_xy,
            r=1000
        )
    ]
)

print("\n" + "=" * 70)
print("5. 새 피처와 원본 전체 기록 비교")

distance_match = np.abs(
    raw_nearest
    - merged["dist_subway_m"].to_numpy()
) <= 1

count_500_match = (
    raw_500
    == merged["subway_500"].to_numpy()
)

count_1000_match = (
    raw_1000
    == merged["subway_1000"].to_numpy()
)

print(
    "최근접 거리 일치(1m 이내):",
    int(distance_match.sum()),
    "/",
    len(merged)
)

print(
    "500m 개수 일치:",
    int(count_500_match.sum()),
    "/",
    len(merged)
)

print(
    "1000m 개수 일치:",
    int(count_1000_match.sum()),
    "/",
    len(merged)
)

print("\n주의:")
print("새 피처는 가까운 동일 역명 좌표를 통합했으므로")
print("원본 1099행을 그대로 센 개수와 다를 수 있습니다.")

print("\n" + "=" * 70)
print("6. 기존 1000m 개수와 새 개수 비교")

merged["count_diff"] = (
    merged["subway_1000"]
    - merged["old_subway_1000"]
)

summary = (
    merged.groupby("city")
    .agg(
        stations=("station", "size"),
        different=(
            "count_diff",
            lambda x: int((x != 0).sum())
        ),
        mean_old=("old_subway_1000", "mean"),
        mean_new=("subway_1000", "mean"),
        max_abs_diff=(
            "count_diff",
            lambda x: x.abs().max()
        )
    )
)

print(summary.round(3).to_string())

print("\n" + "=" * 70)
print("7. 개수 차이 상위 20개")

examples = merged.copy()
examples["abs_count_diff"] = (
    examples["count_diff"].abs()
)

print(
    examples.sort_values(
        "abs_count_diff",
        ascending=False
    )
    .head(20)[
        [
            "city",
            "station",
            "old_subway_1000",
            "subway_1000",
            "count_diff"
        ]
    ]
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("8. 기본 논리 검증")

invalid_counts = (
    (merged["subway_500"] < 0)
    | (merged["subway_1000"] < 0)
    | (
        merged["subway_500"]
        > merged["subway_1000"]
    )
)

print(
    "음수 또는 반경 역전:",
    int(invalid_counts.sum())
)

if invalid_counts.any():
    raise ValueError("지하철 개수 논리 오류")

print("\n지하철 개수 검증 완료!")
print("파일 저장 및 수정 없음")
