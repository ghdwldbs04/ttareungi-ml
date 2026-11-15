
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

cols = [
    "lat",
    "lon",
    "cycleway_m_500",
    "dist_cycleway_m"
]

for col in cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

cw = df[df["city"] == "changwon"].copy()

print("=" * 70)
print("1. 창원 대여소 기본 정보")

print("전체 대여소:", len(cw))

print(
    "반경 500m 내 자전거도로 길이 > 0:",
    (cw["cycleway_m_500"] > 0).sum()
)

print(
    "반경 500m 내 자전거도로 길이 = 0:",
    (cw["cycleway_m_500"] == 0).sum()
)

print(
    "가장 가까운 자전거도로 거리 = 5000m:",
    (cw["dist_cycleway_m"] == 5000).sum()
)

print("\n" + "=" * 70)
print("2. 자전거도로가 잡힌 대여소 전체 목록")

positive = cw[
    cw["cycleway_m_500"] > 0
].copy()

show_cols = [
    "station",
    "lat",
    "lon",
    "cycleway_m_500",
    "dist_cycleway_m"
]

print(
    positive[show_cols]
    .sort_values(
        "cycleway_m_500",
        ascending=False
    )
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("3. 자전거도로가 잡힌 대여소의 좌표 범위")

for name, group in [
    ("전체 창원", cw),
    ("자전거도로 길이 > 0", positive),
    ("자전거도로 길이 = 0", cw[cw["cycleway_m_500"] == 0])
]:
    print(f"\n[{name}]")
    print("대여소 수:", len(group))

    if len(group) == 0:
        continue

    print(
        "위도 범위:",
        round(group["lat"].min(), 6),
        "~",
        round(group["lat"].max(), 6)
    )

    print(
        "경도 범위:",
        round(group["lon"].min(), 6),
        "~",
        round(group["lon"].max(), 6)
    )

print("\n" + "=" * 70)
print("4. 거리 구간별 대여소 수")

bins = [
    -np.inf,
    100,
    300,
    500,
    1000,
    2000,
    4000,
    5000,
    np.inf
]

labels = [
    "0~100m",
    "100~300m",
    "300~500m",
    "500~1000m",
    "1000~2000m",
    "2000~4000m",
    "4000~5000m",
    "5000m 초과"
]

cw["distance_group"] = pd.cut(
    cw["dist_cycleway_m"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(
    cw["distance_group"]
    .value_counts(sort=False)
    .to_string()
)

print("\n" + "=" * 70)
print("5. 자전거도로 거리 최솟값·최댓값 대여소")

print("\n가장 가까운 대여소 10개:")
print(
    cw[show_cols]
    .sort_values("dist_cycleway_m")
    .head(10)
    .round(3)
    .to_string(index=False)
)

print("\n가장 먼 대여소 10개:")
print(
    cw[show_cols]
    .sort_values(
        "dist_cycleway_m",
        ascending=False
    )
    .head(10)
    .round(3)
    .to_string(index=False)
)

print("\n창원 자전거도로 공간 분포 검사 완료!")
