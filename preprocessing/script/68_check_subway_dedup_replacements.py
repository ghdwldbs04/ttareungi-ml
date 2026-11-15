
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
print("2. 좌표 변환")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

rx, ry = transformer.transform(
    raw["역경도"].to_numpy(),
    raw["역위도"].to_numpy()
)

raw = raw.copy()
raw["x"] = rx
raw["y"] = ry

merged = stations[
    ["city", "station", "lat", "lon"]
].merge(
    features[
        ["city", "station", "dist_subway_m"]
    ],
    on=["city", "station"],
    how="left",
    validate="one_to_one"
)

sx, sy = transformer.transform(
    merged["lon"].to_numpy(),
    merged["lat"].to_numpy()
)

station_xy = np.column_stack([sx, sy])
raw_xy = np.column_stack([rx, ry])

tree = cKDTree(raw_xy)

raw_dist, raw_idx = tree.query(
    station_xy,
    k=1
)

merged["raw_dist"] = raw_dist
merged["raw_idx"] = raw_idx

merged["diff_m"] = (
    merged["dist_subway_m"]
    - merged["raw_dist"]
)

different = merged[
    merged["diff_m"] > 1
].copy()

print("새 거리가 더 긴 대여소:", len(different))

print("\n" + "=" * 70)
print("3. 같은 역명의 다른 좌표 확인")

results = []

for _, station in different.iterrows():
    nearest_raw = raw.iloc[
        int(station["raw_idx"])
    ]

    name = nearest_raw["역사명"]

    same_name = raw[
        raw["역사명"] == name
    ].copy()

    x, y = transformer.transform(
        station["lon"],
        station["lat"]
    )

    same_name["station_dist"] = np.sqrt(
        (same_name["x"] - x) ** 2
        + (same_name["y"] - y) ** 2
    )

    # 원본 최근접 역사와 다른 좌표가
    # 100m 이내에 존재하는지 확인
    nearest_x = nearest_raw["x"]
    nearest_y = nearest_raw["y"]

    same_name["pair_dist"] = np.sqrt(
        (same_name["x"] - nearest_x) ** 2
        + (same_name["y"] - nearest_y) ** 2
    )

    nearby = same_name[
        (same_name["pair_dist"] > 0.001)
        & (same_name["pair_dist"] <= 100)
    ].copy()

    if len(nearby):
        best = nearby.sort_values(
            "station_dist"
        ).iloc[0]

        replacement_name = best["역사명"]
        replacement_line = best["노선명"]
        replacement_dist = best["station_dist"]
        pair_dist = best["pair_dist"]
    else:
        replacement_name = ""
        replacement_line = ""
        replacement_dist = np.nan
        pair_dist = np.nan

    results.append({
        "city": station["city"],
        "station": station["station"],
        "raw_nearest_name": name,
        "raw_nearest_line": nearest_raw["노선명"],
        "raw_dist": station["raw_dist"],
        "new_dist": station["dist_subway_m"],
        "diff_m": station["diff_m"],
        "same_name_count": len(same_name),
        "replacement_line": replacement_line,
        "replacement_dist": replacement_dist,
        "pair_dist": pair_dist,
        "replacement_vs_new_error": (
            abs(
                replacement_dist
                - station["dist_subway_m"]
            )
            if pd.notna(replacement_dist)
            else np.nan
        ),
    })

result = pd.DataFrame(results)

print("\n" + "=" * 70)
print("4. 통합 후보 좌표 확인 결과")

print(
    "100m 이내 동일 역명 다른 좌표 존재:",
    int(result["pair_dist"].notna().sum()),
    "/",
    len(result)
)

print(
    "대체 좌표 거리와 새 거리 1m 이내 일치:",
    int(
        (
            result["replacement_vs_new_error"]
            <= 1
        ).sum()
    ),
    "/",
    len(result)
)

print("\n" + "=" * 70)
print("5. 차이가 큰 대여소 상위 25개")

print(
    result.sort_values(
        "diff_m",
        ascending=False
    )
    .head(25)
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("6. 대체 좌표가 없거나 새 거리와 불일치한 경우")

unexplained = result[
    result["replacement_vs_new_error"].isna()
    | (
        result["replacement_vs_new_error"]
        > 1
    )
]

print("추가 확인 필요:", len(unexplained))

if len(unexplained):
    print(
        unexplained.sort_values(
            "diff_m",
            ascending=False
        )
        .head(20)
        .round(3)
        .to_string(index=False)
    )

print("\n지하철 좌표 통합 영향 최종 검사 완료!")
print("파일 저장 및 수정 없음")
