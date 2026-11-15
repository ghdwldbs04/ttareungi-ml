
from pathlib import Path
import unicodedata

import geopandas as gpd
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

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
print("1. 격자 SHP 파일 확인")

print("SHP 파일 수:", len(shp_files))

if not shp_files:
    raise FileNotFoundError(
        "격자 SHP 파일이 없습니다."
    )

print("\n" + "=" * 70)
print("2. 대표 SHP 구조 확인")

sample = shp_files[0]

print("파일:", nfc(sample.relative_to(BASE)))

gdf = gpd.read_file(sample)

print("행 수:", len(gdf))
print("열 이름:", list(gdf.columns))
print("좌표계:", gdf.crs)

print("\n자료형:")
print(gdf.dtypes.astype(str).to_string())

print("\n앞 5행:")
print(
    gdf.drop(columns="geometry")
    .head(5)
    .to_string(index=False)
)

print("\n도형 종류:")
print(
    gdf.geometry.geom_type
    .value_counts()
    .to_string()
)

print("\n경계 좌표 범위:")
print(gdf.total_bounds)

print("\n" + "=" * 70)
print("3. 전체 SHP의 좌표계와 격자 수")

results = []

for path in shp_files:
    grid = gpd.read_file(path)

    results.append({
        "file": nfc(path.stem),
        "rows": len(grid),
        "crs": str(grid.crs),
        "invalid_geometry": int(
            (~grid.geometry.is_valid).sum()
        ),
        "empty_geometry": int(
            grid.geometry.is_empty.sum()
        ),
        "missing_geometry": int(
            grid.geometry.isna().sum()
        )
    })

result_df = pd.DataFrame(results)

print(
    result_df.to_string(index=False)
)

print("\n전체 격자 행 수:", result_df["rows"].sum())

print("\n좌표계별 파일 수:")
print(
    result_df["crs"]
    .value_counts()
    .to_string()
)

print(
    "\n유효하지 않은 도형 합계:",
    result_df["invalid_geometry"].sum()
)

print(
    "빈 도형 합계:",
    result_df["empty_geometry"].sum()
)

print(
    "결측 도형 합계:",
    result_df["missing_geometry"].sum()
)

print("\nSGIS 격자 경계 구조 검사 완료!")
