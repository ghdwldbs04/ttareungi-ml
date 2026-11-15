
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

files = [
    p for p in FOOTAGE.rglob("*.csv")
    if "전국자전거도로표준데이터"
    in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"자전거도로 원본 파일 확인 필요: {files}")

df = pd.read_csv(
    files[0],
    encoding="cp949",
    low_memory=False
)

# 1. 도시 구분
sido = df["시도명"].fillna("").astype(str)
sigungu = df["시군구명"].fillna("").astype(str)

conditions = [
    sido.str.contains("서울", regex=False),
    sido.str.contains("대전", regex=False),
    sido.str.contains("세종", regex=False),
    sigungu.str.contains("창원", regex=False),
]

choices = [
    "seoul",
    "daejeon",
    "sejong",
    "changwon"
]

df["city"] = np.select(
    conditions,
    choices,
    default="other"
)

target = df[
    df["city"] != "other"
].copy()

print("=" * 65)
print("1. 연구 대상 도시 자전거도로 기록 수")

print("전국 전체:", len(df))
print("연구 대상 도시:", len(target))
print("\n도시별 기록 수:")
print(target["city"].value_counts().to_string())

# 2. 좌표 검사
coord_cols = [
    "기점위도",
    "기점경도",
    "종점위도",
    "종점경도"
]

for col in coord_cols:
    target[col] = pd.to_numeric(
        target[col],
        errors="coerce"
    )

start_valid = (
    target["기점위도"].between(33, 39)
    & target["기점경도"].between(124, 132)
)

end_valid = (
    target["종점위도"].between(33, 39)
    & target["종점경도"].between(124, 132)
)

target["coord_status"] = np.select(
    [
        start_valid & end_valid,
        start_valid & ~end_valid,
        ~start_valid & end_valid
    ],
    [
        "both",
        "start_only",
        "end_only"
    ],
    default="neither"
)

print("\n" + "=" * 65)
print("2. 도시별 좌표 보유 상태")

status_table = pd.crosstab(
    target["city"],
    target["coord_status"]
)

print(status_table.to_string())

print("\n좌표 상태 전체:")
print(target["coord_status"].value_counts().to_string())

# 3. 도로 길이 검사
target["length_km"] = pd.to_numeric(
    target["총길이(km)"],
    errors="coerce"
)

print("\n" + "=" * 65)
print("3. 도로 길이 검사")

print("길이 결측:", target["length_km"].isna().sum())
print("길이 0 이하:", (target["length_km"] <= 0).sum())

print("\n도시별 길이 통계 (km):")
print(
    target.groupby("city")["length_km"]
    .agg(["count", "mean", "median", "max", "sum"])
    .round(2)
    .to_string()
)

# 4. 자전거도로 종류
print("\n" + "=" * 65)
print("4. 도시별 자전거도로 종류")

print(
    pd.crosstab(
        target["city"],
        target["자전거도로종류"].fillna("결측")
    ).to_string()
)

# 5. 제공기관 확인
print("\n" + "=" * 65)
print("5. 도시별 제공기관")

for city, group in target.groupby("city"):
    print(f"\n[{city}]")
    print(
        group["제공기관명"]
        .value_counts()
        .head(10)
        .to_string()
    )

print("\n자전거도로 연구 대상 도시 검사 완료!")
