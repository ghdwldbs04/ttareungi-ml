
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

grids = []

for path in shp_files:
    gdf = gpd.read_file(path)

    if gdf.crs.to_epsg() != 5179:
        raise ValueError(
            f"좌표계 오류: {path.name}"
        )

    grids.append(
        gdf[["GRID_CD", "geometry"]]
    )

grid_all = gpd.GeoDataFrame(
    pd.concat(grids, ignore_index=True),
    geometry="geometry",
    crs="EPSG:5179"
)

grid_all["GRID_CD"] = (
    grid_all["GRID_CD"].astype(str).str.strip()
)

if grid_all["GRID_CD"].duplicated().any():
    raise ValueError("격자코드 중복 발견")

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
    grid_all,
    how="left",
    predicate="intersects"
)

if (
    len(matched) != len(stations)
    or matched["GRID_CD"].isna().any()
    or matched["_row_id"].duplicated().any()
):
    raise ValueError(
        "격자 연결 결과가 51번 검사와 다릅니다."
    )

result = matched[
    ["_row_id", "city", "station", "GRID_CD"]
].copy()

print("대여소 수:", len(result))
print("고유 격자 수:", result["GRID_CD"].nunique())

print("\n" + "=" * 70)
print("2. 핵심 통계항목 읽기")

all_csv = list(STAT.rglob("*.csv"))

for feature, (topic, code) in FEATURE_CODES.items():
    files = sorted(
        [
            p for p in all_csv
            if f"_{topic}_" in nfc(p.name)
        ],
        key=lambda p: nfc(str(p))
    )

    pieces = []
    source_rows = 0

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

        source_rows += len(selected)

        if selected.empty:
            continue

        selected["격자코드"] = (
            selected["격자코드"].str.strip()
        )

        selected[feature] = pd.to_numeric(
            selected["통계값"].str.strip(),
            errors="coerce"
        )

        pieces.append(
            selected[["격자코드", feature]]
        )

    if not pieces:
        print(f"\n[{feature}] 해당 통계 없음")
        result[feature] = float("nan")
        continue

    combined = pd.concat(
        pieces,
        ignore_index=True
    )

    duplicates = combined[
        "격자코드"
    ].duplicated().sum()

    print(f"\n[{feature}]")
    print("통계 코드:", code)
    print("CSV 파일 수:", len(files))
    print("추출 행 수:", source_rows)
    print("고유 격자 수:", combined["격자코드"].nunique())
    print("중복 격자코드 행 수:", duplicates)
    print(
        "통계값 결측:",
        combined[feature].isna().sum()
    )
    print(
        "음수 통계값:",
        (combined[feature] < 0).sum()
    )

    if duplicates:
        raise ValueError(
            f"{feature}: 격자코드 중복"
        )

    result = result.merge(
        combined.rename(
            columns={"격자코드": "GRID_CD"}
        ),
        on="GRID_CD",
        how="left",
        validate="many_to_one"
    )

print("\n" + "=" * 70)
print("3. 대여소별 핵심 통계 연결 결과")

feature_cols = list(FEATURE_CODES)

print("결과 행 수:", len(result))
print(
    "대여소 키 중복:",
    result.duplicated(["city", "station"]).sum()
)

print("\n전체 결측 수:")
print(
    result[feature_cols]
    .isna()
    .sum()
    .to_string()
)

print("\n도시별 결측 비율 (%):")
print(
    (
        result.groupby("city")[feature_cols]
        .apply(lambda x: x.isna().mean() * 100)
    )
    .round(2)
    .to_string()
)

print("\n" + "=" * 70)
print("4. 통계값 분포")

print(
    result[feature_cols]
    .describe()
    .round(2)
    .to_string()
)

print("\n" + "=" * 70)
print("5. 다섯 통계가 모두 없는 대여소")

all_missing = result[
    result[feature_cols].isna().all(axis=1)
]

print("전체:", len(all_missing))

print("\n도시별:")
print(
    all_missing["city"]
    .value_counts()
    .to_string()
)

print("\n예시 최대 20개:")
print(
    all_missing[
        ["city", "station", "GRID_CD"]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nSGIS 핵심 통계 연결 검사 완료!")
