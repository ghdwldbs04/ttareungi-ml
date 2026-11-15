
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

# 1. 기존 데이터와 새 대학 피처 불러오기
old = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

new = pd.read_csv(
    PROCESSED / "university_features_new.csv",
    dtype={"station": str}
)

features = [
    "dist_univ_m",
    "univ_300",
    "univ_500",
    "univ_1000"
]

print("=" * 60)
print("1. 기존 데이터의 대학 관련 열")

related = [
    col for col in old.columns
    if "univ" in col.lower()
    or "university" in col.lower()
    or "대학" in col
]

print("발견한 열:", related)

print("\n" + "=" * 60)
print("2. 대여소 연결 검증")

merged = old.merge(
    new,
    on=["city", "station"],
    how="outer",
    suffixes=("_old", "_new"),
    indicator=True,
    validate="one_to_one"
)

print(merged["_merge"].value_counts().to_string())

print("\n" + "=" * 60)
print("3. 기존 대학 피처와 새 피처 비교")

for feature in features:
    if feature not in old.columns:
        print(f"\n[{feature}]")
        print("기존 데이터에 없는 새 피처입니다.")
        continue

    old_col = f"{feature}_old"
    new_col = f"{feature}_new"

    old_values = pd.to_numeric(
        merged[old_col], errors="coerce"
    )
    new_values = pd.to_numeric(
        merged[new_col], errors="coerce"
    )

    valid = old_values.notna() & new_values.notna()

    diff = new_values.loc[valid] - old_values.loc[valid]

    tolerance = 0.01 if feature == "dist_univ_m" else 0

    match = np.isclose(
        diff,
        0,
        atol=tolerance,
        rtol=0
    )

    print(f"\n[{feature}]")
    print("비교 가능한 대여소:", valid.sum())
    print("일치하는 대여소:", match.sum())

    if valid.sum() > 0:
        print("일치율:", round(match.mean() * 100, 2), "%")
        print("평균 절대 차이:", diff.abs().mean())
        print("최대 절대 차이:", diff.abs().max())

        examples = merged.loc[
            valid,
            ["city", "station", old_col, new_col]
        ].copy()

        examples["abs_diff"] = diff.abs()

        print("\n차이가 가장 큰 대여소 5개")
        print(
            examples.sort_values(
                "abs_diff", ascending=False
            ).head(5).to_string(index=False)
        )

print("\n대학교 피처 비교 완료!")
