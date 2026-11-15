
from pathlib import Path
import unicodedata

import geopandas as gpd
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
STATIONS = BASE / "processed" / "stations_clean.csv"

def nfc(value):
    return unicodedata.normalize("NFC", str(value))

sgis_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and "SGIS" in nfc(p.name).upper()
]

if len(sgis_dirs) != 1:
    raise ValueError(
        f"SGIS 폴더 확인 필요: {len(sgis_dirs)}"
    )

SGIS = sgis_dirs[0]

shp_files = sorted(
    SGIS.rglob("*.shp"),
    key=lambda p: nfc(str(p))
)

print("=" * 70)
print("1. SGIS 격자 합치기")

grids = []

for path in shp_files:
    grid = gpd.read_file(path)

    if grid.crs is None:
        raise ValueError(
            f"좌표계 정보 없음: {path.name}"
        )

    if grid.crs.to_epsg() != 5179:
        raise ValueError(
            f"예상과 다른 좌표계: {path.name}, {grid.crs}"
        )

    grids.append(
        grid[["GRID_CD", "geometry"]]
    )

all_grids = gpd.GeoDataFrame(
    pd.concat(grids, ignore_index=True),
    geometry="geometry",
    crs="EPSG:5179"
)

all_grids["GRID_CD"] = (
    all_grids["GRID_CD"]
    .astype(str)
    .str.strip()
)

print("전체 격자 수:", len(all_grids))
print(
    "격자코드 중복 행 수:",
    all_grids["GRID_CD"].duplicated().sum()
)
print(
    "고유 격자코드 수:",
    all_grids["GRID_CD"].nunique()
)

if all_grids["GRID_CD"].duplicated().any():
    print("\n중복 격자코드 예시:")
    print(
        all_grids.loc[
            all_grids["GRID_CD"].duplicated(
                keep=False
            ),
            ["GRID_CD"]
        ]
        .head(20)
        .to_string(index=False)
    )

print("\n" + "=" * 70)
print("2. 대여소 좌표 확인")

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

required = [
    "city",
    "station",
    "lat",
    "lon"
]

missing = [
    col for col in required
    if col not in stations.columns
]

if missing:
    raise ValueError(
        f"대여소 필수 열 누락: {missing}"
    )

stations["lat"] = pd.to_numeric(
    stations["lat"],
    errors="coerce"
)

stations["lon"] = pd.to_numeric(
    stations["lon"],
    errors="coerce"
)

print("전체 대여소:", len(stations))

print(
    "좌표 결측 대여소:",
    stations[["lat", "lon"]]
    .isna()
    .any(axis=1)
    .sum()
)

print(
    "도시+대여소 중복:",
    stations.duplicated(
        ["city", "station"]
    ).sum()
)

if stations[["lat", "lon"]].isna().any().any():
    raise ValueError(
        "대여소 좌표에 결측값이 있습니다."
    )

# 원본 행 순서를 보존할 식별자
stations = stations.copy()
stations["_row_id"] = range(len(stations))

station_points = gpd.GeoDataFrame(
    stations[
        ["_row_id", "city", "station"]
    ].copy(),
    geometry=gpd.points_from_xy(
        stations["lon"],
        stations["lat"]
    ),
    crs="EPSG:4326"
)

station_points = station_points.to_crs(
    "EPSG:5179"
)

print("\n" + "=" * 70)
print("3. 대여소가 속한 격자 찾기")

matched = gpd.sjoin(
    station_points,
    all_grids,
    how="left",
    predicate="intersects"
)

# 경계선에 있는 대여소는
# 둘 이상의 격자와 만날 수 있음
match_counts = (
    matched.groupby("_row_id")["GRID_CD"]
    .count()
)

print(
    "격자에 연결되지 않은 대여소:",
    (match_counts == 0).sum()
)

print(
    "정확히 1개 격자에 연결된 대여소:",
    (match_counts == 1).sum()
)

print(
    "2개 이상 격자에 연결된 대여소:",
    (match_counts >= 2).sum()
)

print(
    "공간 조인 결과 전체 행 수:",
    len(matched)
)

print("\n" + "=" * 70)
print("4. 도시별 연결 현황")

status = station_points[
    ["_row_id", "city", "station"]
].copy()

status["grid_match_count"] = (
    status["_row_id"]
    .map(match_counts)
    .fillna(0)
    .astype(int)
)

status["status"] = "one_grid"

status.loc[
    status["grid_match_count"] == 0,
    "status"
] = "no_grid"

status.loc[
    status["grid_match_count"] >= 2,
    "status"
] = "multiple_grids"

summary = pd.crosstab(
    status["city"],
    status["status"]
)

print(summary.to_string())

print("\n" + "=" * 70)
print("5. 연결되지 않은 대여소 예시")

unmatched = status[
    status["status"] == "no_grid"
]

if unmatched.empty:
    print("없음")
else:
    print(
        unmatched[
            ["city", "station", "grid_match_count"]
        ]
        .head(20)
        .to_string(index=False)
    )

print("\n" + "=" * 70)
print("6. 여러 격자에 연결된 대여소 예시")

multiple = status[
    status["status"] == "multiple_grids"
]

if multiple.empty:
    print("없음")
else:
    print(
        multiple[
            ["city", "station", "grid_match_count"]
        ]
        .head(20)
        .to_string(index=False)
    )

print("\nSGIS 격자 공간 연결 검사 완료!")
