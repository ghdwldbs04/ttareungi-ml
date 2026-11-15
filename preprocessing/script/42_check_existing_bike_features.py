
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

required = ["city", "station"] + features

missing_cols = [
    col for col in required
    if col not in df.columns
]

if missing_cols:
    raise ValueError(f"필요한 열이 없음: {missing_cols}")

print("=" * 70)
print("1. 기본 정보")

print("전체 대여소 수:", len(df))
print("\n도시별 대여소 수:")
print(df["city"].value_counts().to_string())

print("\n기존 자전거도로 피처:")
print(features)

# 숫자 변환
for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\n" + "=" * 70)
print("2. 피처별 결측·음수·무한대 검사")

for col in features:
    values = df[col]

    print(f"\n[{col}]")
    print("결측:", values.isna().sum())
    print("음수:", (values < 0).sum())
    print("무한대:", np.isinf(values).sum())
    print("0인 대여소:", (values == 0).sum())
    print("고유값 수:", values.nunique(dropna=True))

print("\n" + "=" * 70)
print("3. 도시별 피처 통계")

for col in features:
    print(f"\n[{col}]")

    summary = df.groupby("city")[col].agg(
        ["count", "mean", "median", "min", "max"]
    )

    print(summary.round(3).to_string())

print("\n" + "=" * 70)
print("4. 도시별 0값 비율 (%)")

zero_rates = (
    df.groupby("city")[features]
    .apply(lambda x: (x == 0).mean() * 100)
)

print(zero_rates.round(2).to_string())

print("\n" + "=" * 70)
print("5. 두 길이 피처의 단위 관계")

# bike_km_500을 m 단위로 변환해 비교
km_as_m = df["bike_km_500"] * 1000
cycle_m = df["cycleway_m_500"]

difference = (km_as_m - cycle_m).abs()

print(
    "두 값이 거의 같은 대여소 수 (오차 1m 이내):",
    (difference <= 1).sum()
)

print(
    "전체 대여소 수:",
    len(df)
)

print(
    "평균 절대 차이 (m):",
    round(difference.mean(), 3)
)

print(
    "최대 절대 차이 (m):",
    round(difference.max(), 3)
)

print("\n" + "=" * 70)
print("6. 기존 자전거도로 피처 상관관계")

print(
    df[features]
    .corr()
    .round(3)
    .to_string()
)

print("\n" + "=" * 70)
print("7. 도시별 피처 예시")

for city, group in df.groupby("city"):
    print(f"\n[{city}]")

    print(
        group[
            ["station"] + features
        ]
        .head(5)
        .to_string(index=False)
    )

print("\n기존 자전거도로 피처 검사 완료!")
