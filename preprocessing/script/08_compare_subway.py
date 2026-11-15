
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "processed"

# 1. 기존 데이터와 새 데이터 읽기
old = pd.read_csv(
    OUT_DIR / "stations_clean.csv",
    dtype={"station": "string"}
)

new = pd.read_csv(
    OUT_DIR / "subway_features_new.csv",
    dtype={"station": "string"}
)

# 2. 대여소 ID 기준 연결
merged = old.merge(
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

# 3. 기존 값과 새 값 비교
print("\n" + "=" * 55)
print("2. 지하철 피처 비교")

for feature in ["dist_subway_m", "subway_1000"]:
    old_col = feature + "_old"
    new_col = feature + "_new"

    a = pd.to_numeric(merged[old_col], errors="coerce")
    b = pd.to_numeric(merged[new_col], errors="coerce")

    valid = a.notna() & b.notna()
    diff = b[valid] - a[valid]

    print(f"\n[{feature}]")
    print("비교 가능한 대여소:", valid.sum())
    print("값이 일치하는 대여소:", np.isclose(
        a[valid], b[valid], atol=0.01, rtol=0
    ).sum())
    print("평균 차이(새 값 - 기존 값):", diff.mean())
    print("평균 절대 차이:", diff.abs().mean())
    print("최대 절대 차이:", diff.abs().max())
    print("상관계수:", a[valid].corr(b[valid]))

# 4. 도시별 거리 비교
print("\n" + "=" * 55)
print("3. 도시별 지하철 거리 평균")

print(
    merged.groupby("city")[
        ["dist_subway_m_old", "dist_subway_m_new"]
    ].mean().round(1).to_string()
)

# 5. 차이가 큰 대여소 확인
merged["distance_diff"] = (
    merged["dist_subway_m_new"]
    - merged["dist_subway_m_old"]
).abs()

print("\n" + "=" * 55)
print("4. 지하철 거리 차이가 큰 대여소 10개")

print(
    merged.sort_values(
        "distance_diff", ascending=False
    )[
        [
            "city",
            "station",
            "dist_subway_m_old",
            "dist_subway_m_new",
            "distance_diff"
        ]
    ].head(10).to_string(index=False)
)

print("\n지하철 피처 비교 완료!")
