
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

universities = pd.read_csv(
    PROCESSED / "universities_clean.csv"
)

features = pd.read_csv(
    PROCESSED / "university_features_new.csv",
    dtype={"station": str}
)

feature_cols = [
    "dist_univ_m",
    "univ_300",
    "univ_500",
    "univ_1000"
]

print("=" * 60)
print("1. 행 수 및 대여소 연결 확인")

print("기준 대여소 수:", len(stations))
print("대학 캠퍼스 위치 수:", len(universities))
print("대학 피처 행 수:", len(features))

print(
    "기준 대여소 중복:",
    stations.duplicated(["city", "station"]).sum()
)

print(
    "피처 대여소 중복:",
    features.duplicated(["city", "station"]).sum()
)

merged = stations[["city", "station"]].merge(
    features[["city", "station"]],
    on=["city", "station"],
    how="outer",
    indicator=True,
    validate="one_to_one"
)

print("연결 결과:")
print(merged["_merge"].value_counts().to_string())

print("\n" + "=" * 60)
print("2. 대학 피처 값 검증")

print("결측값:")
print(features[feature_cols].isna().sum().to_string())

print(
    "음수 거리:",
    (features["dist_univ_m"] < 0).sum()
)

for col in ["univ_300", "univ_500", "univ_1000"]:
    values = features[col]

    invalid = (
        (values < 0)
        | (values % 1 != 0)
    )

    print(f"{col} 잘못된 개수 값:", invalid.sum())

print(
    "반경별 개수 순서 오류:",
    (
        (features["univ_300"] > features["univ_500"])
        | (features["univ_500"] > features["univ_1000"])
    ).sum()
)

print("\n" + "=" * 60)
print("3. 거리와 반경별 개수의 논리적 관계")

for radius, col in [
    (300, "univ_300"),
    (500, "univ_500"),
    (1000, "univ_1000"),
]:
    # 가장 가까운 대학이 반경 안에 있으면
    # 해당 반경의 대학 개수는 반드시 1 이상
    error_inside = (
        (features["dist_univ_m"] < radius)
        & (features[col] == 0)
    ).sum()

    # 가장 가까운 대학이 반경 밖에 있으면
    # 해당 반경의 대학 개수는 반드시 0
    error_outside = (
        (features["dist_univ_m"] > radius)
        & (features[col] > 0)
    ).sum()

    print(
        f"{radius}m: "
        f"반경 내부인데 개수 0 = {error_inside}, "
        f"반경 외부인데 개수 있음 = {error_outside}"
    )

print("\n" + "=" * 60)
print("4. 도시별 대학 피처 요약")

print(
    features.groupby("city")[feature_cols]
    .agg(["mean", "median", "max"])
    .round(2)
    .to_string()
)

print("\n대학교 피처 최종 검증 완료!")
