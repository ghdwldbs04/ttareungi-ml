
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
print("2. 조사 대상 12개 대여소")

target_ids = [
    "558",
    "3561",
    "6057",
    "1277",
    "2175",
    "4027",
    "525",
    "397",
    "3003",
    "220",
    "338",
    "222",
]

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

targets = merged[
    (merged["city"] == "seoul")
    & (merged["station"].isin(target_ids))
].copy()

print("검사 대상:", len(targets))

if len(targets) != 12:
    raise ValueError("검사 대상 대여소 누락")

print("\n" + "=" * 70)
print("3. 원본 역사 및 대여소 좌표 변환")

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

raw_xy = np.column_stack([rx, ry])
tree = cKDTree(raw_xy)

print("\n" + "=" * 70)
print("4. 대여소별 주변 역사 거리 비교")

results = []

for _, station in targets.iterrows():
    x, y = transformer.transform(
        station["lon"],
        station["lat"]
    )

    new_dist = float(
        station["dist_subway_m"]
    )

    # 새 거리보다 약간 넓은 범위의
    # 모든 원본 역사 후보 조사
    candidate_idx = tree.query_ball_point(
        [x, y],
        r=new_dist + 100
    )

    candidates = raw.iloc[
        candidate_idx
    ].copy()

    candidates["distance_m"] = np.sqrt(
        (candidates["x"] - x) ** 2
        + (candidates["y"] - y) ** 2
    )

    candidates["new_error_m"] = (
        candidates["distance_m"]
        - new_dist
    ).abs()

    candidates = candidates.sort_values(
        "distance_m"
    )

    matching = candidates[
        candidates["new_error_m"] <= 1
    ].sort_values("new_error_m")

    if len(matching):
        best = matching.iloc[0]

        matched_name = best["역사명"]
        matched_line = best["노선명"]
        matched_dist = best["distance_m"]
        matched_error = best["new_error_m"]
    else:
        matched_name = ""
        matched_line = ""
        matched_dist = np.nan
        matched_error = np.nan

    results.append({
        "station": station["station"],
        "new_dist": new_dist,
        "matching_name": matched_name,
        "matching_line": matched_line,
        "matching_dist": matched_dist,
        "matching_error": matched_error,
        "nearby_raw_records": len(candidates),
    })

    print("\n" + "-" * 65)
    print("대여소:", station["station"])
    print("새 거리:", round(new_dist, 3))

    print("가까운 원본 역사 상위 8개:")

    print(
        candidates[
            [
                "역사명",
                "노선명",
                "distance_m",
                "new_error_m"
            ]
        ]
        .head(8)
        .round(3)
        .to_string(index=False)
    )

    print(
        "새 거리와 1m 이내 일치하는 역사:",
        len(matching)
    )

print("\n" + "=" * 70)
print("5. 전체 결과 요약")

result = pd.DataFrame(results)

print(
    result.sort_values(
        "station"
    )
    .round(3)
    .to_string(index=False)
)

print("\n새 거리와 일치하는 역사 발견:")
print(
    int(result["matching_error"].notna().sum()),
    "/",
    len(result)
)

print("\n새 거리와 일치하는 역사 미발견:")
print(
    int(result["matching_error"].isna().sum())
)

print("\n남은 지하철 거리 사례 조사 완료!")
print("파일 저장 및 수정 없음")
