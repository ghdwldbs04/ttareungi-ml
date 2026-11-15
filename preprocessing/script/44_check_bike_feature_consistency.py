
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PATH = BASE / "processed" / "stations_clean.csv"

df = pd.read_csv(
    PATH,
    dtype={"station": str},
    low_memory=False
)

features = [
    "bike_pts_500",
    "bike_km_500",
    "cycleway_m_500",
    "dist_cycleway_m"
]

for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# 1. 반경 500m와 최단거리 관계
dist = df["dist_cycleway_m"]
length = df["cycleway_m_500"]

# 경계값 반올림 오차를 피하기 위해 1m 여유
far_but_length = (
    (dist > 501)
    & (length > 0)
)

near_but_zero = (
    (dist < 499)
    & (length == 0)
)

df["far_but_length"] = far_but_length
df["near_but_zero"] = near_but_zero

print("=" * 70)
print("1. 거리와 반경 500m 길이의 관계")

print(
    "최단거리 > 501m인데 길이 > 0:",
    far_but_length.sum()
)

print(
    "최단거리 < 499m인데 길이 = 0:",
    near_but_zero.sum()
)

print("\n도시별 불일치 후보:")
print(
    df.groupby("city")[
        ["far_but_length", "near_but_zero"]
    ]
    .sum()
    .to_string()
)

# 2. 거리 5000m 상한 검사
print("\n" + "=" * 70)
print("2. 최단거리 5000m 상한 검사")

print(
    "정확히 5000m인 대여소:",
    (dist == 5000).sum()
)

print(
    "5000m 초과:",
    (dist > 5000).sum()
)

print("\n도시별 정확히 5000m인 대여소:")
print(
    df.assign(at_5000=dist == 5000)
    .groupby("city")["at_5000"]
    .sum()
    .to_string()
)

# 3. 두 자전거도로 피처 그룹 비교
print("\n" + "=" * 70)
print("3. 기존 두 피처 그룹의 0값 관계")

pts = df["bike_pts_500"]
km = df["bike_km_500"]

print(
    "bike_pts_500 = 0, bike_km_500 > 0:",
    ((pts == 0) & (km > 0)).sum()
)

print(
    "bike_pts_500 > 0, bike_km_500 = 0:",
    ((pts > 0) & (km == 0)).sum()
)

print(
    "bike_pts_500 = 0, cycleway_m_500 > 0:",
    ((pts == 0) & (length > 0)).sum()
)

print(
    "bike_pts_500 > 0, cycleway_m_500 = 0:",
    ((pts > 0) & (length == 0)).sum()
)

# 4. 불일치 후보 예시
print("\n" + "=" * 70)
print("4. 거리·길이 불일치 후보 예시")

cols = [
    "city",
    "station",
    "dist_cycleway_m",
    "cycleway_m_500",
    "bike_pts_500",
    "bike_km_500"
]

print("\n[최단거리 > 501m인데 길이 > 0]")
print(
    df.loc[far_but_length, cols]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\n[최단거리 < 499m인데 길이 = 0]")
print(
    df.loc[near_but_zero, cols]
    .head(10)
    .round(2)
    .to_string(index=False)
)

# 5. 도시별 비율
print("\n" + "=" * 70)
print("5. 도시별 불일치 후보 비율 (%)")

rates = (
    df.groupby("city")[
        ["far_but_length", "near_but_zero"]
    ]
    .mean() * 100
)

print(rates.round(2).to_string())

print("\n기존 자전거도로 피처 논리 검증 완료!")
    