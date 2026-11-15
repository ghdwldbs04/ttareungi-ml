
from pathlib import Path
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "footage"
OUT_DIR = BASE_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)

# 1. 대여소 기준 데이터 읽기
stations = pd.read_csv(
    OUT_DIR / "stations_clean.csv",
    encoding="utf-8-sig",
    dtype={"station": "string"}
)

# 2. 버스정류장 원본 자동 찾기
required = ["정류장번호", "위도", "경도"]
matches = []

for path in DATA_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() != ".csv":
        continue

    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            sample = pd.read_csv(path, encoding=enc, nrows=3)
            if all(c in sample.columns for c in required):
                matches.append((path, enc))
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            break

if len(matches) != 1:
    raise ValueError(f"버스정류장 파일 검색 결과: {len(matches)}개")

bus_path, encoding = matches[0]

bus = pd.read_csv(
    bus_path,
    encoding=encoding,
    low_memory=False,
    dtype={"정류장번호": "string"}
)

print("버스정류장 원본:", bus_path.name)
print("전체 버스정류장:", len(bus))
print("전체 대여소:", len(stations))

# 3. 좌표를 숫자로 변환하고 이상값 제거
for col in ["위도", "경도"]:
    bus[col] = pd.to_numeric(bus[col], errors="coerce")

bus = bus[
    bus["위도"].between(33, 39)
    & bus["경도"].between(124, 132)
].copy()

# 4. 위도·경도를 미터 단위 좌표로 변환
# EPSG:5179 = 한국 통합 좌표계
station_geo = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326"
).to_crs("EPSG:5179")

bus_geo = gpd.GeoDataFrame(
    bus,
    geometry=gpd.points_from_xy(bus["경도"], bus["위도"]),
    crs="EPSG:4326"
).to_crs("EPSG:5179")

station_xy = list(zip(station_geo.geometry.x, station_geo.geometry.y))
bus_xy = list(zip(bus_geo.geometry.x, bus_geo.geometry.y))

# 5. 빠른 공간 검색
tree = cKDTree(bus_xy)

result = stations[["city", "station"]].copy()

for radius in [300, 500, 1000]:
    counts = [
        len(indices)
        for indices in tree.query_ball_point(station_xy, r=radius)
    ]

    result[f"bus_{radius}"] = counts
    print(f"반경 {radius}m 계산 완료")

# 6. 결과 검사
print("\n피처 요약")
print(result[["bus_300", "bus_500", "bus_1000"]].describe())

# 반경이 커지면 개수가 줄어들면 안 됨
assert (result["bus_300"] <= result["bus_500"]).all()
assert (result["bus_500"] <= result["bus_1000"]).all()

# 7. 별도 CSV로 저장
output_path = OUT_DIR / "bus_features_new.csv"
result.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n저장 완료:", output_path)
print("저장된 행 수:", len(result))
print("\n처음 10개 대여소:")
print(result.head(10).to_string(index=False))
