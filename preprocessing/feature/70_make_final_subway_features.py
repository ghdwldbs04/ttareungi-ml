
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

OUTPUT = PROCESSED / "subway_features_final.csv"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. 파일 찾기 및 읽기")

stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str},
    low_memory=False
)

features = pd.read_csv(
    PROCESSED / "subway_features_new.csv",
    dtype={"station": str},
    low_memory=False
)

sources = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "도시철도역사정보" in nfc(p.name)
]

if len(sources) != 1:
    raise ValueError("지하철 원본 파일 확인 필요")

raw = pd.read_excel(sources[0])

print("대여소:", len(stations))
print("기존 새 피처:", len(features))
print("원본 역사:", len(raw))

if OUTPUT.exists():
    raise FileExistsError(
        f"이미 파일이 존재합니다: {OUTPUT}"
    )

print("\n" + "=" * 70)
print("2. 기본 데이터 검증")

if len(stations) != 5224:
    raise ValueError("대여소 행 수 확인 필요")

if len(features) != 5224:
    raise ValueError("새 지하철 피처 행 수 확인 필요")

for name, df in [
    ("대여소", stations),
    ("지하철 피처", features)
]:
    if df.duplicated(["city", "station"]).any():
        raise ValueError(f"{name}: 대여소 키 중복")

if raw[["역위도", "역경도"]].isna().any().any():
    raise ValueError("원본 역사 좌표 결측")

print("기본 검증: 정상")

print("\n" + "=" * 70)
print("3. 원본 전체 역사 기준 최근접 거리 계산")

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

sx, sy = transformer.transform(
    stations["lon"].to_numpy(),
    stations["lat"].to_numpy()
)

rx, ry = transformer.transform(
    raw["역경도"].to_numpy(),
    raw["역위도"].to_numpy()
)

station_xy = np.column_stack([sx, sy])
raw_xy = np.column_stack([rx, ry])

tree = cKDTree(raw_xy)

nearest_dist, _ = tree.query(
    station_xy,
    k=1
)

print("거리 계산 완료:", len(nearest_dist))
print("최소 거리:", round(float(nearest_dist.min()), 3))
print("최대 거리:", round(float(nearest_dist.max()), 3))

print("\n" + "=" * 70)
print("4. 최종 지하철 피처 구성")

distance_df = stations[
    ["city", "station"]
].copy()

distance_df["dist_subway_m"] = nearest_dist

count_df = features[
    [
        "city",
        "station",
        "subway_500",
        "subway_1000"
    ]
].copy()

final = distance_df.merge(
    count_df,
    on=["city", "station"],
    how="left",
    validate="one_to_one",
    indicator=True
)

if not (final["_merge"] == "both").all():
    raise ValueError("피처 병합 누락")

final = final.drop(columns="_merge")

print("최종 행 수:", len(final))
print("최종 열:", final.columns.tolist())

print("\n" + "=" * 70)
print("5. 저장 전 검증")

if final.isna().any().any():
    raise ValueError("최종 피처 결측 발생")

if final.duplicated(["city", "station"]).any():
    raise ValueError("대여소 키 중복")

if not final[
    ["city", "station"]
].equals(
    stations[["city", "station"]]
):
    raise ValueError("대여소 키 또는 순서 불일치")

if (
    final["subway_500"]
    > final["subway_1000"]
).any():
    raise ValueError("지하철 개수 반경 논리 오류")

if (final["dist_subway_m"] < 0).any():
    raise ValueError("음수 거리 발생")

if not np.isfinite(
    final["dist_subway_m"].to_numpy()
).all():
    raise ValueError("비정상 거리 발생")

print("행 수: 정상")
print("대여소 키 및 순서: 정상")
print("결측: 없음")
print("거리 및 개수 범위: 정상")

print("\n" + "=" * 70)
print("6. 최종 피처 저장")

final.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("저장 파일:", OUTPUT.relative_to(BASE))
print("저장 행 수:", len(final))
print("저장 열 수:", len(final.columns))

print("\n도시별 평균 거리 및 개수:")

print(
    final.groupby("city")[
        [
            "dist_subway_m",
            "subway_500",
            "subway_1000"
        ]
    ]
    .mean()
    .round(3)
    .to_string()
)

print("\n최종 지하철 피처 생성 완료!")
print("기존 파일 수정 없음")
