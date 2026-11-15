
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

# 1. 데이터 불러오기
stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

universities = pd.read_csv(
    PROCESSED / "universities_clean.csv"
)

print("대여소 수:", len(stations))
print("대학 캠퍼스 위치 수:", len(universities))

# 2. 필수 열 확인
station_columns = ["city", "station", "lat", "lon"]
university_columns = ["위도", "경도"]

for col in station_columns:
    if col not in stations.columns:
        raise ValueError(f"대여소 데이터에 {col} 열이 없습니다.")

for col in university_columns:
    if col not in universities.columns:
        raise ValueError(f"대학교 데이터에 {col} 열이 없습니다.")

# 3. 좌표 결측 확인
if stations[["lat", "lon"]].isna().any().any():
    raise ValueError("대여소 좌표에 결측값이 있습니다.")

if universities[["위도", "경도"]].isna().any().any():
    raise ValueError("대학교 좌표에 결측값이 있습니다.")

# 4. 위경도 → EPSG:5179 변환
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

station_x, station_y = transformer.transform(
    stations["lon"].to_numpy(),
    stations["lat"].to_numpy()
)

univ_x, univ_y = transformer.transform(
    universities["경도"].to_numpy(),
    universities["위도"].to_numpy()
)

station_xy = np.column_stack(
    [station_x, station_y]
)

univ_xy = np.column_stack(
    [univ_x, univ_y]
)

# 5. 대학 캠퍼스 위치 검색용 KDTree
tree = cKDTree(univ_xy)

# 6. 가장 가까운 대학 캠퍼스까지 거리
distances, _ = tree.query(
    station_xy,
    k=1
)

# 7. 반경별 대학 캠퍼스 위치 수
counts_300 = tree.query_ball_point(
    station_xy, r=300, return_length=True
)

counts_500 = tree.query_ball_point(
    station_xy, r=500, return_length=True
)

counts_1000 = tree.query_ball_point(
    station_xy, r=1000, return_length=True
)

# 8. 결과 데이터프레임
result = stations[["city", "station"]].copy()

result["dist_univ_m"] = distances
result["univ_300"] = counts_300
result["univ_500"] = counts_500
result["univ_1000"] = counts_1000

# 9. 기본 검증
print("\n" + "=" * 55)
print("대학교 피처 요약")

features = [
    "dist_univ_m",
    "univ_300",
    "univ_500",
    "univ_1000"
]

print(result[features].describe().to_string())

print("\n반경별 대학 수 관계 확인")

invalid_order = (
    (result["univ_300"] > result["univ_500"])
    | (result["univ_500"] > result["univ_1000"])
)

print("반경별 대학 수 순서 이상:", invalid_order.sum())

print("\n피처 결측값")
print(result[features].isna().sum().to_string())

print("\n처음 10개 대여소")
print(result.head(10).to_string(index=False))

# 10. 결과 저장
output = PROCESSED / "university_features_new.csv"

result.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 위치:", output)
print("대학교 피처 생성 완료!")
