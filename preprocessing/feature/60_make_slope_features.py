
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

STATIONS = PROCESSED / "stations_clean.csv"
OUTPUT = PROCESSED / "slope_features_new.csv"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. 원본 파일 찾기")

slope_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and (
        "Slope" in nfc(p.name)
        or "경사" in nfc(p.name)
    )
]

if len(slope_dirs) != 1:
    raise ValueError("경사도 폴더 확인 필요")

candidates = [
    p for p in slope_dirs[0].rglob("*.tif")
    if nfc(p.name).lower() == "slope_all.tif"
]

if len(candidates) != 1:
    raise ValueError("Slope_All.tif 확인 필요")

RASTER = candidates[0]

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

if len(stations) != 5224:
    raise ValueError("대여소 수 확인 필요")

if stations[["city", "station"]].isna().any().any():
    raise ValueError("대여소 키 결측")

if stations.duplicated(["city", "station"]).any():
    raise ValueError("대여소 키 중복")

if stations[["lat", "lon"]].isna().any().any():
    raise ValueError("대여소 좌표 결측")

print("대여소:", len(stations))
print("래스터:", nfc(RASTER.relative_to(BASE)))

print("\n" + "=" * 70)
print("2. 전체 대여소 경사도 추출")

with rasterio.open(RASTER) as src:
    if src.crs is None:
        raise ValueError("래스터 좌표계 없음")

    xs, ys = transform(
        "EPSG:4326",
        src.crs,
        stations["lon"].tolist(),
        stations["lat"].tolist()
    )

    coords = list(zip(xs, ys))

    values = []

    for item in src.sample(
        coords,
        indexes=1,
        masked=True
    ):
        value = item[0]

        if np.ma.is_masked(value):
            values.append(np.nan)
        else:
            values.append(float(value))

features = stations[
    ["city", "station"]
].copy()

features["slope_value"] = values

features["slope_missing"] = (
    features["slope_value"]
    .isna()
    .astype(int)
)

print("추출 행 수:", len(features))
print(
    "경사도 결측:",
    int(features["slope_missing"].sum())
)
print(
    "경사도 0:",
    int((features["slope_value"] == 0).sum())
)

print("\n" + "=" * 70)
print("3. 저장 전 검증")

if len(features) != len(stations):
    raise ValueError("행 수 불일치")

if features.duplicated(["city", "station"]).any():
    raise ValueError("피처 키 중복")

if not features[
    ["city", "station"]
].equals(
    stations[["city", "station"]]
):
    raise ValueError("대여소 키 또는 순서 불일치")

if int(features["slope_missing"].sum()) != 7:
    raise ValueError("59번과 결측 수 불일치")

if (
    features["slope_value"]
    .dropna()
    .lt(0)
    .any()
):
    raise ValueError("음수 경사도 발견")

if not np.isfinite(
    features["slope_value"].dropna()
).all():
    raise ValueError("비정상 경사도 값 발견")

print("행 수: 정상")
print("대여소 키 및 순서: 정상")
print("결측 수: 정상")
print("경사도 값 범위: 정상")

print("\n" + "=" * 70)
print("4. 피처 파일 저장")

if OUTPUT.exists():
    raise FileExistsError(
        f"기존 파일이 있어 덮어쓰지 않습니다: {OUTPUT}"
    )

features.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("저장 파일:", OUTPUT.relative_to(BASE))
print("저장 행 수:", len(features))
print("저장 열 수:", len(features.columns))

print("\n도시별 결측 수:")
print(
    features.groupby("city")["slope_missing"]
    .sum()
    .to_string()
)

print("\n경사도 피처 파일 생성 완료!")
print("기존 stations_clean.csv 수정 없음")
