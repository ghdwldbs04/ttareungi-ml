
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

# 1. 대여소와 전처리된 공원 데이터 읽기
stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

parks = pd.read_csv(
    PROCESSED / "parks_clean.csv"
)

print("대여소 수:", len(stations))
print("전처리된 공원 수:", len(parks))

# 2. 좌표를 미터 단위로 변환
# EPSG:4326 -> EPSG:5179
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

station_x, station_y = transformer.transform(
    stations["lon"].to_numpy(),
    stations["lat"].to_numpy()
)

park_x, park_y = transformer.transform(
    parks["경도"].to_numpy(),
    parks["위도"].to_numpy()
)

station_xy = np.column_stack([
    station_x, station_y
])

park_xy = np.column_stack([
    park_x, park_y
])

# 3. 공원 위치 검색용 트리
park_tree = cKDTree(park_xy)

# 4. 가장 가까운 공원까지의 거리
distances, indices = park_tree.query(
    station_xy,
    k=1
)

# 5. 반경 500m 공원 검색
nearby_parks = park_tree.query_ball_point(
    station_xy,
    r=500
)

# 6. 공원 면적
park_areas = parks["공원면적"].to_numpy()

park_counts = np.array([
    len(indices)
    for indices in nearby_parks
])

park_area_sums = np.array([
    park_areas[indices].sum()
    for indices in nearby_parks
])

# 7. 결과 저장
result = stations[["city", "station"]].copy()

result["dist_park_m"] = distances
result["park_500"] = park_counts
result["park_area_500"] = park_area_sums

output = PROCESSED / "park_features_new.csv"

result.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n공원 피처 요약")
print(
    result[
        ["dist_park_m", "park_500", "park_area_500"]
    ].describe()
)

print("\n저장 완료:", output)
print("저장된 대여소 수:", len(result))

print("\n처음 10개 대여소")
print(result.head(10).to_string(index=False))

print("\n공원 피처 생성 완료!")
