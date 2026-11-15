
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
    raise ValueError("지하철 원본 엑셀 확인 필요")

raw = pd.read_excel(sources[0])

print("대여소:", len(stations))
print("지하철 피처:", len(features))
print("원본 역사:", len(raw))

print("\n" + "=" * 70)
print("2. 서울 거리 불일치 대여소 추출")

old = stations[
    ["city", "station", "lat", "lon", "dist_subway_m"]
].rename(
    columns={"dist_subway_m": "old_dist"}
)

new = features[
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

merged["abs_diff"] = (
    merged["old_dist"]
    - merged["new_dist"].clip(upper=5000)
).abs()

different = merged[
    (merged["city"] == "seoul")
    & (merged["abs_diff"] > 1)
].copy()

print("서울 불일치 대여소:", len(different))

if len(different) != 241:
    print("주의: 이전 결과와 불일치 개수가 다릅니다.")

print("\n" + "=" * 70)
print("3. 전체 역사 좌표와 대여소 좌표 변환")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

raw_x, raw_y = transformer.transform(
    raw["역경도"].to_numpy(),
    raw["역위도"].to_numpy()
)

station_x, station_y = transformer.transform(
    different["lon"].to_numpy(),
    different["lat"].to_numpy()
)

raw_xy = np.column_stack([raw_x, raw_y])
station_xy = np.column_stack([station_x, station_y])

tree = cKDTree(raw_xy)

print("\n" + "=" * 70)
print("4. 기존 거리와 일치하는 역사 탐색")

results = []

for i, (_, station) in enumerate(different.iterrows()):
    point = station_xy[i]
    old_distance = float(station["old_dist"])

    # 기존 거리 주변의 역사 좌표를 찾음
    candidate_idx = tree.query_ball_point(
        point,
        r=old_distance + 10
    )

    if candidate_idx:
        candidate_xy = raw_xy[candidate_idx]

        candidate_dist = np.sqrt(
            np.sum(
                (candidate_xy - point) ** 2,
                axis=1
            )
        )

        candidate_error = np.abs(
            candidate_dist - old_distance
        )

        best_pos = int(
            np.argmin(candidate_error)
        )

        best_idx = candidate_idx[best_pos]
        best_dist = candidate_dist[best_pos]
        best_error = candidate_error[best_pos]

        matched_name = raw.iloc[
            best_idx
        ]["역사명"]

        matched_line = raw.iloc[
            best_idx
        ]["노선명"]
    else:
        matched_name = ""
        matched_line = ""
        best_dist = np.nan
        best_error = np.nan

    results.append({
        "station": station["station"],
        "old_dist": old_distance,
        "new_dist": station["new_dist"],
        "abs_diff": station["abs_diff"],
        "matched_name": matched_name,
        "matched_line": matched_line,
        "matched_dist": best_dist,
        "matching_error": best_error,
    })

result = pd.DataFrame(results)

print("분석 대여소:", len(result))

for tolerance in [0.01, 1, 5, 10]:
    count = int(
        (result["matching_error"] <= tolerance)
        .sum()
    )

    print(
        f"기존 거리와 {tolerance}m 이내 일치:",
        count,
        "/",
        len(result)
    )

print("\n" + "=" * 70)
print("5. 기존 거리와 일치하는 역사 상위 빈도")

matched = result[
    result["matching_error"] <= 1
]

print("1m 이내 일치 대여소:", len(matched))

if len(matched):
    print(
        matched.groupby(
            ["matched_name", "matched_line"]
        )
        .size()
        .sort_values(ascending=False)
        .head(25)
        .to_string()
    )

print("\n" + "=" * 70)
print("6. 거리 차이 상위 25개 상세")

print(
    result.sort_values(
        "abs_diff",
        ascending=False
    )
    .head(25)
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("7. 기존 거리와 일치하는 역을 찾지 못한 사례")

unmatched = result[
    result["matching_error"].isna()
    | (result["matching_error"] > 1)
]

print("1m 이내 일치 역사 없음:", len(unmatched))

if len(unmatched):
    print(
        unmatched.sort_values(
            "abs_diff",
            ascending=False
        )
        .head(20)
        .round(3)
        .to_string(index=False)
    )

print("\n기존 지하철 거리 추적 완료!")
print("파일 저장 및 수정 없음")
