
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
print("1. 원본 파일 읽기")

stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str},
    low_memory=False
)

subway = pd.read_csv(
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
print("새 지하철 피처:", len(subway))
print("역사 원본:", len(raw))

print("\n" + "=" * 70)
print("2. 거리 차이 상위 25개 선택")

old = stations[
    ["city", "station", "lat", "lon", "dist_subway_m"]
].rename(
    columns={"dist_subway_m": "old_dist"}
)

new = subway[
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

top25 = (
    merged[merged["city"] == "seoul"]
    .sort_values("abs_diff", ascending=False)
    .head(25)
    .copy()
)

print("검사 대상:", len(top25))

print("\n" + "=" * 70)
print("3. 모든 역 좌표를 미터 좌표계로 변환")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

raw_x, raw_y = transformer.transform(
    raw["역경도"].to_numpy(),
    raw["역위도"].to_numpy()
)

raw = raw.copy()
raw["x"] = raw_x
raw["y"] = raw_y

station_x, station_y = transformer.transform(
    top25["lon"].to_numpy(),
    top25["lat"].to_numpy()
)

tree = cKDTree(
    np.column_stack([raw["x"], raw["y"]])
)

nearest_dist, nearest_idx = tree.query(
    np.column_stack([station_x, station_y]),
    k=1
)

print("\n" + "=" * 70)
print("4. 같은 역명의 노선별 거리 비교")

results = []

for i, (_, station) in enumerate(top25.iterrows()):
    nearest = raw.iloc[nearest_idx[i]]
    station_name = nearest["역사명"]

    same_name = raw[
        raw["역사명"] == station_name
    ].copy()

    distances = np.sqrt(
        (same_name["x"].to_numpy() - station_x[i]) ** 2
        + (same_name["y"].to_numpy() - station_y[i]) ** 2
    )

    same_name["distance_m"] = distances

    same_name["old_diff_m"] = (
        same_name["distance_m"]
        - station["old_dist"]
    ).abs()

    closest_to_old = same_name.sort_values(
        "old_diff_m"
    ).iloc[0]

    results.append({
        "station": station["station"],
        "nearest_name": station_name,
        "nearest_line": nearest["노선명"],
        "old_dist": station["old_dist"],
        "new_dist": station["new_dist"],
        "old_matching_line": closest_to_old["노선명"],
        "old_matching_distance": closest_to_old["distance_m"],
        "old_matching_error": closest_to_old["old_diff_m"],
        "same_name_count": len(same_name),
    })

result = pd.DataFrame(results)

print(
    result.round(3).to_string(index=False)
)

print("\n" + "=" * 70)
print("5. 기존 거리와 같은 역의 다른 좌표 일치 여부")

for tolerance in [1, 5, 10]:
    count = int(
        (result["old_matching_error"] <= tolerance)
        .sum()
    )

    print(
        f"오차 {tolerance}m 이내:",
        count,
        "/",
        len(result)
    )

print("\n" + "=" * 70)
print("6. 주요 사례의 노선별 거리 상세")

selected_stations = [
    "370",
    "2608",
    "1674",
    "3009",
    "1206",
]

for station_id in selected_stations:
    matches = top25[
        top25["station"] == station_id
    ]

    if matches.empty:
        continue

    station = matches.iloc[0]

    x, y = transformer.transform(
        station["lon"],
        station["lat"]
    )

    nearest_distance, nearest_index = tree.query(
        [x, y],
        k=1
    )

    station_name = raw.iloc[
        nearest_index
    ]["역사명"]

    same_name = raw[
        raw["역사명"] == station_name
    ].copy()

    same_name["distance_m"] = np.sqrt(
        (same_name["x"] - x) ** 2
        + (same_name["y"] - y) ** 2
    )

    print("\n" + "-" * 60)
    print("대여소:", station_id)
    print("최근접 역명:", station_name)
    print("기존 거리:", round(station["old_dist"], 3))
    print("새 거리:", round(station["new_dist"], 3))

    print(
        same_name[
            ["역사명", "노선명", "distance_m"]
        ]
        .sort_values("distance_m")
        .round(3)
        .to_string(index=False)
    )

print("\n환승역 좌표 처리 방식 검사 완료!")
print("파일 저장 및 수정 없음")
