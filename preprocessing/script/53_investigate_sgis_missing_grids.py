
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
    raise ValueError("SGIS 폴더 확인 필요")

SGIS = sgis_dirs[0]
STAT = SGIS / "1. 통계"

FEATURE_CODES = {
    "grid_pop_total": ("인구", "to_in_001"),
    "grid_households": ("가구", "to_ga_001"),
    "grid_housing": ("주택", "to_ho_001"),
    "grid_businesses": ("사업체", "to_fa_010"),
    "grid_workers": ("종사자", "to_em_020"),
}

print("=" * 70)
print("1. 대여소와 격자 연결")

shp_files = sorted(
    SGIS.rglob("*.shp"),
    key=lambda p: nfc(str(p))
)

grid_parts = []

for path in shp_files:
    gdf = gpd.read_file(path)

    if gdf.crs is None or gdf.crs.to_epsg() != 5179:
        raise ValueError(f"좌표계 확인 필요: {path.name}")

    grid_parts.append(
        gdf[["GRID_CD", "geometry"]]
    )

grids = gpd.GeoDataFrame(
    pd.concat(grid_parts, ignore_index=True),
    geometry="geometry",
    crs="EPSG:5179"
)

grids["GRID_CD"] = (
    grids["GRID_CD"].astype(str).str.strip()
)

if grids["GRID_CD"].duplicated().any():
    raise ValueError("격자코드 중복")

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

stations["_row_id"] = range(len(stations))

points = gpd.GeoDataFrame(
    stations[["_row_id", "city", "station"]].copy(),
    geometry=gpd.points_from_xy(
        stations["lon"],
        stations["lat"]
    ),
    crs="EPSG:4326"
).to_crs("EPSG:5179")

matched = gpd.sjoin(
    points,
    grids,
    how="left",
    predicate="intersects"
)

if (
    len(matched) != len(stations)
    or matched["GRID_CD"].isna().any()
    or matched["_row_id"].duplicated().any()
):
    raise ValueError("대여소 격자 연결 오류")

station_grid = matched[
    ["_row_id", "city", "station", "GRID_CD"]
].copy()

print("대여소:", len(station_grid))
print("사용 격자:", station_grid["GRID_CD"].nunique())

print("\n" + "=" * 70)
print("2. 다섯 통계의 격자별 존재 여부")

all_csv = list(STAT.rglob("*.csv"))

grid_stats = (
    station_grid[["GRID_CD"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

for feature, (topic, code) in FEATURE_CODES.items():
    files = [
        p for p in all_csv
        if f"_{topic}_" in nfc(p.name)
    ]

    pieces = []

    for path in files:
        df = pd.read_csv(
            path,
            encoding="cp949",
            dtype=str,
            usecols=[
                "기준연도",
                "격자코드",
                "통계항목",
                "통계값"
            ]
        )

        selected = df[
            (df["기준연도"].str.strip() == "2024")
            & (df["통계항목"].str.strip() == code)
        ].copy()

        if selected.empty:
            continue

        selected["격자코드"] = (
            selected["격자코드"].str.strip()
        )

        selected[feature] = pd.to_numeric(
            selected["통계값"],
            errors="coerce"
        )

        pieces.append(
            selected[["격자코드", feature]]
        )

    if not pieces:
        raise ValueError(f"통계 없음: {feature}")

    combined = pd.concat(
        pieces,
        ignore_index=True
    ).rename(columns={"격자코드": "GRID_CD"})

    if combined["GRID_CD"].duplicated().any():
        raise ValueError(f"중복 격자: {feature}")

    grid_stats = grid_stats.merge(
        combined,
        on="GRID_CD",
        how="left",
        validate="one_to_one"
    )

features = list(FEATURE_CODES)

grid_stats["missing_count"] = (
    grid_stats[features].isna().sum(axis=1)
)

print("전체 사용 격자:", len(grid_stats))

print("\n누락된 통계항목 수별 격자 개수:")
print(
    grid_stats["missing_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n항목별 결측 격자 수:")
print(
    grid_stats[features]
    .isna()
    .sum()
    .to_string()
)

print("\n" + "=" * 70)
print("3. 결측이 있는 격자의 대여소 현황")

missing_grids = grid_stats[
    grid_stats["missing_count"] > 0
].copy()

affected = station_grid.merge(
    missing_grids,
    on="GRID_CD",
    how="inner"
)

print("결측 격자 수:", len(missing_grids))
print("영향받는 대여소 수:", len(affected))

print("\n도시별 영향받는 대여소:")
print(
    affected["city"]
    .value_counts()
    .to_string()
)

print("\n결측 격자별 상세 정보:")
detail = (
    affected.groupby(["city", "GRID_CD"])
    .agg(
        station_count=("station", "size"),
        stations=(
            "station",
            lambda x: ", ".join(x.head(5))
        )
    )
    .reset_index()
    .merge(
        missing_grids,
        on="GRID_CD",
        how="left",
        validate="many_to_one"
    )
)

display_cols = [
    "city",
    "GRID_CD",
    "station_count",
    "missing_count",
    *features,
    "stations"
]

print(
    detail[display_cols]
    .sort_values(
        ["city", "missing_count", "GRID_CD"],
        ascending=[True, False, True]
    )
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("4. 결측 격자 주변의 통계 존재 여부")

# 주변 격자와 닿아 있는지 확인한다.
# 결측값을 대체하거나 저장하지는 않는다.
missing_geo = grids.merge(
    missing_grids[["GRID_CD", "missing_count"]],
    on="GRID_CD",
    how="inner",
    validate="one_to_one"
)

neighbor_pairs = gpd.sjoin(
    missing_geo[["GRID_CD", "geometry"]],
    grids[["GRID_CD", "geometry"]],
    how="left",
    predicate="touches",
    lsuffix="missing",
    rsuffix="neighbor"
)

neighbor_counts = (
    neighbor_pairs
    .groupby("GRID_CD_missing")["GRID_CD_neighbor"]
    .nunique()
)

print("결측 격자별 접촉 격자 수:")
print(
    neighbor_counts
    .describe()
    .round(2)
    .to_string()
)

print("\n접촉 격자가 없는 결측 격자:")
no_neighbors = missing_grids[
    ~missing_grids["GRID_CD"].isin(
        neighbor_counts[neighbor_counts > 0].index
    )
]

if no_neighbors.empty:
    print("없음")
else:
    print(
        no_neighbors["GRID_CD"].to_string(index=False)
    )

print("\nSGIS 결측 격자 조사 완료!")
