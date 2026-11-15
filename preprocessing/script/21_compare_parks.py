
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

# 기존 데이터
old = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

# 새로 만든 공원 피처
new = pd.read_csv(
    PROCESSED / "park_features_new.csv",
    dtype={"station": str}
)

features = [
    "dist_park_m",
    "park_500",
    "park_area_500"
]

# 대여소 기준으로 연결
merged = old[
    ["city", "station"] + features
].merge(
    new,
    on=["city", "station"],
    how="outer",
    suffixes=("_old", "_new"),
    indicator=True,
    validate="one_to_one"
)

print("=" * 55)
print("1. 대여소 연결 결과")
print(merged["_merge"].value_counts().to_string())

print("\n" + "=" * 55)
print("2. 기존 공원 피처와 새 피처 비교")

for feature in features:
    old_col = f"{feature}_old"
    new_col = f"{feature}_new"

    valid = merged[
        [old_col, new_col]
    ].notna().all(axis=1)

    diff = (
        merged.loc[valid, new_col]
        - merged.loc[valid, old_col]
    )

    # 거리와 면적은 소수점 오차를 허용
    tolerance = 0.01 if feature != "park_500" else 0

    match = np.isclose(
        diff,
        0,
        atol=tolerance,
        rtol=0
    )

    print(f"\n[{feature}]")
    print("비교 가능한 대여소:", valid.sum())
    print("값이 일치하는 대여소:", match.sum())
    print("일치율:", round(match.mean() * 100, 2), "%")
    print("평균 차이(새 값 - 기존 값):", diff.mean())
    print("평균 절대 차이:", diff.abs().mean())
    print("최대 절대 차이:", diff.abs().max())

    print("\n도시별 일치율")

    temp = merged.loc[valid, ["city"]].copy()
    temp["match"] = match

    print(
        temp.groupby("city")["match"]
        .agg(["count", "sum", "mean"])
        .to_string()
    )

    # 차이가 큰 대여소 5개
    examples = merged.loc[
        valid,
        ["city", "station", old_col, new_col]
    ].copy()

    examples["abs_diff"] = diff.abs()

    print("\n차이가 가장 큰 대여소 5개")
    print(
        examples.sort_values(
            "abs_diff",
            ascending=False
        ).head(5).to_string(index=False)
    )

print("\n공원 피처 비교 완료!")
