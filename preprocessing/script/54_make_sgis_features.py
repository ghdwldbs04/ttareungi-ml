
from pathlib import Path
import unicodedata

import geopandas as gpd
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

STATIONS = PROCESSED / "stations_clean.csv"
OUTPUT = PROCESSED / "sgis_features_new.csv"


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
print("1. SGIS 격자 읽기")

shp_files = sorted(
    SGIS.rglob("*.shp"),
    key=lambda p: nfc(str(p))
)

parts = []

for path in shp_files:
    gdf = gpd.read_file(path)

    if gdf.crs is None or gdf.crs.to_epsg() != 5179:
        raise ValueError(
            f"좌표계 확인 필요: {path.name}"
        )

    parts.append(
        gdf[["GRID_CD", "geometry"]]
    )

grids = gpd.GeoDataFrame(
    pd.concat(parts, ignore_index=True),
    geometry="geometry",
    crs="EPSG:5179"
)

grids["GRID_CD"] = (
    grids["GRID_CD"]
    .astype(str)
    .str.strip()
)

if grids["GRID_CD"].duplicated().any():
    raise ValueError("격자코드 중복")

print("격자 수:", len(grids))

print("\n" + "=" * 70)
print("2. 대여소와 격자 연결")

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

stations["_row_id"] = range(len(stations))

points = gpd.GeoDataFrame(
    stations[
        ["_row_id", "city", "station"]
    ].copy(),
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
    raise ValueError(
        "대여소와 격자 연결 결과 확인 필요"
    )

result = matched[
    ["_row_id", "city", "station", "GRID_CD"]
].copy()

print("대여소 수:", len(result))
print("사용 격자 수:", result["GRID_CD"].nunique())

print("\n" + "=" * 70)
print("3. 핵심 통계 연결")

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
            selected["통계값"].str.strip(),
            errors="coerce"
        )

        pieces.append(
            selected[["격자코드", feature]]
        )

    if not pieces:
        raise ValueError(
            f"통계 데이터 없음: {feature}"
        )

    combined = pd.concat(
        pieces,
        ignore_index=True
    ).rename(
        columns={"격자코드": "GRID_CD"}
    )

    if combined["GRID_CD"].duplicated().any():
        raise ValueError(
            f"통계 격자 중복: {feature}"
        )

    if combined[feature].isna().any():
        raise ValueError(
            f"원본 통계값 변환 실패: {feature}"
        )

    result = result.merge(
        combined,
        on="GRID_CD",
        how="left",
        validate="many_to_one"
    )

    print(
        f"{feature}: "
        f"연결 {result[feature].notna().sum()}개, "
        f"결측 {result[feature].isna().sum()}개"
    )

print("\n" + "=" * 70)
print("4. 결측 표시 피처 추가")

feature_cols = list(FEATURE_CODES)

for feature in feature_cols:
    missing_col = f"{feature}_missing"

    result[missing_col] = (
        result[feature]
        .isna()
        .astype(int)
    )

result["grid_any_missing"] = (
    result[feature_cols]
    .isna()
    .any(axis=1)
    .astype(int)
)

result["grid_all_missing"] = (
    result[feature_cols]
    .isna()
    .all(axis=1)
    .astype(int)
)

print(
    "통계 하나 이상 결측:",
    result["grid_any_missing"].sum()
)

print(
    "통계 전부 결측:",
    result["grid_all_missing"].sum()
)

print("\n" + "=" * 70)
print("5. 최종 검증")

result = result.sort_values(
    "_row_id"
).reset_index(drop=True)

if len(result) != len(stations):
    raise ValueError("대여소 행 수 불일치")

if result.duplicated(
    ["city", "station"]
).any():
    raise ValueError("대여소 키 중복")

if not (
    result["city"].to_numpy()
    == stations["city"].to_numpy()
).all():
    raise ValueError("도시 순서 불일치")

if not (
    result["station"].to_numpy()
    == stations["station"].to_numpy()
).all():
    raise ValueError("대여소 순서 불일치")

expected_missing = {
    "grid_pop_total": 26,
    "grid_households": 29,
    "grid_housing": 56,
    "grid_businesses": 11,
    "grid_workers": 11,
}

for feature, expected in expected_missing.items():
    actual = int(result[feature].isna().sum())

    if actual != expected:
        raise ValueError(
            f"{feature} 결측 수 변경: "
            f"예상 {expected}, 실제 {actual}"
        )

if result["grid_any_missing"].sum() != 56:
    raise ValueError(
        "결측 대여소 수가 53번과 다릅니다."
    )

if result["grid_all_missing"].sum() != 11:
    raise ValueError(
        "전체 결측 대여소 수가 52번과 다릅니다."
    )

output_cols = [
    "city",
    "station",
    "GRID_CD",
    *feature_cols,
    *[
        f"{feature}_missing"
        for feature in feature_cols
    ],
    "grid_any_missing",
    "grid_all_missing",
]

output = result[output_cols].copy()

if OUTPUT.exists():
    raise FileExistsError(
        f"기존 파일을 덮어쓰지 않습니다: {OUTPUT}"
    )

output.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("저장 파일:", OUTPUT.relative_to(BASE))
print("저장 행 수:", len(output))
print("저장 열 수:", len(output.columns))

print("\n도시별 행 수:")
print(
    output["city"]
    .value_counts()
    .to_string()
)

print("\nSGIS 피처 생성 완료!")
