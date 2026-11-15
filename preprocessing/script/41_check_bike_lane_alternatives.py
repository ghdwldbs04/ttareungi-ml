
from pathlib import Path
import unicodedata

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

def nfc(text):
    return unicodedata.normalize("NFC", str(text))

print("=" * 70)
print("1. 기존 대여소 데이터의 자전거도로 관련 열")

stations_path = PROCESSED / "stations_clean.csv"

if not stations_path.exists():
    raise FileNotFoundError(stations_path)

stations = pd.read_csv(
    stations_path,
    low_memory=False
)

keywords = [
    "bike",
    "bicycle",
    "cycle",
    "lane",
    "road",
    "자전거",
    "도로"
]

matching_cols = [
    col for col in stations.columns
    if any(
        keyword in nfc(col).lower()
        for keyword in keywords
    )
]

print("전체 대여소 수:", len(stations))
print("전체 열 수:", len(stations.columns))
print("관련 열:", matching_cols)

if matching_cols:
    print("\n관련 열 결측 수:")
    print(stations[matching_cols].isna().sum().to_string())

print("\n" + "=" * 70)
print("2. 원본 폴더의 공간 데이터 파일 형식")

spatial_exts = {
    ".shp", ".shx", ".dbf",
    ".geojson", ".json",
    ".gpkg", ".gdb",
    ".kml", ".kmz",
    ".zip"
}

spatial_files = [
    p for p in FOOTAGE.rglob("*")
    if p.is_file()
    and p.suffix.lower() in spatial_exts
]

ext_counts = {}

for p in spatial_files:
    ext = p.suffix.lower()
    ext_counts[ext] = ext_counts.get(ext, 0) + 1

print("공간 데이터 관련 파일 수:", len(spatial_files))

print("\n확장자별 개수:")
for ext, count in sorted(ext_counts.items()):
    print(f"{ext}: {count}")

print("\n" + "=" * 70)
print("3. 자전거도로 관련 파일명 검색")

bike_keywords = [
    "자전거",
    "bike",
    "bicycle",
    "cycle",
    "도로망"
]

bike_files = [
    p for p in FOOTAGE.rglob("*")
    if p.is_file()
    and any(
        keyword in nfc(p.name).lower()
        for keyword in bike_keywords
    )
]

print("관련 파일 수:", len(bike_files))

for p in bike_files[:50]:
    print(p.relative_to(BASE))

if len(bike_files) > 50:
    print(f"... 외 {len(bike_files) - 50}개")

print("\n" + "=" * 70)
print("4. 공간 데이터 파일 예시")

for p in spatial_files[:30]:
    print(p.relative_to(BASE))

if len(spatial_files) > 30:
    print(f"... 외 {len(spatial_files) - 30}개")

print("\n자전거도로 대체 자료 검사 완료!")
