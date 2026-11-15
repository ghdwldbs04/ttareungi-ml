
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "footage"
OUT_DIR = BASE_DIR / "processed"

# 1. 대여소 데이터
stations = pd.read_csv(
    OUT_DIR / "stations_clean.csv",
    dtype={"station": "string"}
)

# 2. 지하철 엑셀 자동 검색
matches = []

for path in DATA_DIR.rglob("*.xlsx"):
    if path.name.startswith("~$"):
        continue

    try:
        sample = pd.read_excel(path, nrows=3)
        if {"역사명", "역위도", "역경도"}.issubset(sample.columns):
            matches.append(path)
    except Exception:
        continue

if len(matches) != 1:
    raise ValueError(f"지하철 원본 후보: {len(matches)}개")

subway = pd.read_excel(matches[0])
print("지하철 원본 행 수:", len(subway))

# 3. 좌표 정리
subway["역위도"] = pd.to_numeric(subway["역위도"], errors="coerce")
subway["역경도"] = pd.to_numeric(subway["역경도"], errors="coerce")

subway = subway[
    subway["역위도"].between(33, 39)
    & subway["역경도"].between(124, 132)
].copy()

# 4. 미터 단위 좌표계로 변환
subway_geo = gpd.GeoDataFrame(
    subway,
    geometry=gpd.points_from_xy(
        subway["역경도"], subway["역위도"]
    ),
    crs="EPSG:4326"
).to_crs("EPSG:5179")

station_geo = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(
        stations["lon"], stations["lat"]
    ),
    crs="EPSG:4326"
).to_crs("EPSG:5179")

# 5. 같은 이름 + 100m 이내인 역을 하나로 묶기
# 노선별 중복 기록을 제거하기 위한 간단한 기준
subway_geo["x_m"] = subway_geo.geometry.x
subway_geo["y_m"] = subway_geo.geometry.y

subway_geo = subway_geo.sort_values(
    ["역사명", "x_m", "y_m"]
).reset_index(drop=True)

keep_indices = []

for name, group in subway_geo.groupby("역사명"):
    kept_points = []

    for idx, row in group.iterrows():
        xy = np.array([row["x_m"], row["y_m"]])

        if not kept_points:
            keep_indices.append(idx)
            kept_points.append(xy)
            continue

        distances = [
            np.linalg.norm(xy - point)
            for point in kept_points
        ]

        if min(distances) > 100:
            keep_indices.append(idx)
            kept_points.append(xy)

subway_unique = subway_geo.loc[keep_indices].copy()

print("중복 정리 후 역 수:", len(subway_unique))
print("제거된 중복 행 수:", len(subway_geo) - len(subway_unique))

# 6. 거리 계산
station_xy = np.column_stack([
    station_geo.geometry.x,
    station_geo.geometry.y
])

subway_xy = np.column_stack([
    subway_unique.geometry.x,
    subway_unique.geometry.y
])

tree = cKDTree(subway_xy)

distances, _ = tree.query(station_xy, k=1)

result = stations[["city", "station"]].copy()
result["dist_subway_m"] = distances

for radius in [500, 1000]:
    counts = tree.query_ball_point(station_xy, r=radius)
    result[f"subway_{radius}"] = [len(x) for x in counts]

# 7. 결과 검증
assert (result["dist_subway_m"] >= 0).all()
assert (result["subway_500"] <= result["subway_1000"]).all()
assert len(result) == len(stations)

print("\n지하철 피처 요약")
print(
    result[
        ["dist_subway_m", "subway_500", "subway_1000"]
    ].describe().to_string()
)

# 8. 저장
output_path = OUT_DIR / "subway_features_new.csv"

result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:", output_path)
print("저장된 행 수:", len(result))
print("\n처음 10개 대여소")
print(result.head(10).to_string(index=False))
