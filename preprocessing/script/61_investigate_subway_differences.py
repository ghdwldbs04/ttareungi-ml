
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

STATIONS = PROCESSED / "stations_clean.csv"
SUBWAY = PROCESSED / "subway_features_new.csv"


print("=" * 70)
print("1. 원본 및 새 지하철 피처 읽기")

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

subway = pd.read_csv(
    SUBWAY,
    dtype={"station": str},
    low_memory=False
)

print("원본 대여소:", len(stations))
print("새 지하철 피처:", len(subway))

print("\n원본 지하철 관련 열:")
print(
    [
        c for c in stations.columns
        if "subway" in c.lower()
    ]
)

print("\n새 지하철 피처 열:")
print(subway.columns.tolist())

if len(stations) != 5224 or len(subway) != 5224:
    raise ValueError("대여소 수 확인 필요")

for name, df in [
    ("원본", stations),
    ("새 지하철", subway)
]:
    if df[["city", "station"]].isna().any().any():
        raise ValueError(f"{name}: 대여소 키 결측")

    if df.duplicated(["city", "station"]).any():
        raise ValueError(f"{name}: 대여소 키 중복")

if "dist_subway_m" not in stations.columns:
    raise ValueError("기존 dist_subway_m 열 없음")

if "dist_subway_m" not in subway.columns:
    raise ValueError("새 dist_subway_m 열 없음")


print("\n" + "=" * 70)
print("2. 대여소 키 기준 병합")

old = stations[
    ["city", "station", "lat", "lon", "dist_subway_m"]
].rename(
    columns={"dist_subway_m": "old_dist"}
)

new = subway[
    ["city", "station", "dist_subway_m"]
].rename(
    columns={"dist_subway_m": "new_dist"}
)

merged = old.merge(
    new,
    on=["city", "station"],
    how="left",
    validate="one_to_one",
    indicator=True
)

print("병합 결과:")
print(merged["_merge"].value_counts().to_string())

if not (merged["_merge"] == "both").all():
    raise ValueError("지하철 피처 병합 누락")

merged["new_dist_capped"] = merged[
    "new_dist"
].clip(upper=5000)

merged["diff_m"] = (
    merged["old_dist"]
    - merged["new_dist_capped"]
)

merged["abs_diff_m"] = merged["diff_m"].abs()

# 좌표 변환 및 거리 계산에서 발생할 수 있는
# 아주 작은 소수점 오차는 허용
TOLERANCE = 1.0

merged["different"] = (
    merged["abs_diff_m"] > TOLERANCE
)

print("\n" + "=" * 70)
print("3. 도시별 거리 불일치 현황")

summary = (
    merged.groupby("city")
    .agg(
        stations=("station", "size"),
        different=("different", "sum"),
        mean_abs_diff_m=("abs_diff_m", "mean"),
        max_abs_diff_m=("abs_diff_m", "max")
    )
)

summary["different_pct"] = (
    summary["different"]
    / summary["stations"]
    * 100
)

print(summary.round(3).to_string())

print("\n전체 불일치:", int(merged["different"].sum()))

print("\n" + "=" * 70)
print("4. 서울 불일치 거리 분포")

seoul_diff = merged[
    (merged["city"] == "seoul")
    & merged["different"]
].copy()

print("서울 불일치 대여소:", len(seoul_diff))

if len(seoul_diff) > 0:
    print("\n절대 차이 분위수(m):")
    print(
        seoul_diff["abs_diff_m"]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
        .round(3)
        .to_string()
    )

    print("\n기존 거리가 더 큰 경우:")
    print(int((seoul_diff["diff_m"] > 0).sum()))

    print("\n새 거리가 더 큰 경우:")
    print(int((seoul_diff["diff_m"] < 0).sum()))

    
    print("\n차이가 큰 대여소 상위 25개:")

    print(
        seoul_diff
        .sort_values(
            "abs_diff_m",
            ascending=False
        )
        .head(25)[
            [
                "station",
                "lat",
                "lon",
                "old_dist",
                "new_dist",
                "new_dist_capped",
                "diff_m",
                "abs_diff_m"
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


print("\n" + "=" * 70)
print("5. 기존 거리 5,000m 상한 확인")

print(
    "기존 거리 5,000m:",
    int((merged["old_dist"] == 5000).sum())
)

print(
    "새 거리 5,000m 초과:",
    int((merged["new_dist"] > 5000).sum())
)

print(
    "새 거리 결측:",
    int(merged["new_dist"].isna().sum())
)

print("\n서울 지하철 거리 불일치 조사 완료!")
print("파일 저장 및 수정 없음")
