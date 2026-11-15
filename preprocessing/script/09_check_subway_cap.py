
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "processed"

old = pd.read_csv(
    OUT_DIR / "stations_clean.csv",
    dtype={"station": "string"}
)

new = pd.read_csv(
    OUT_DIR / "subway_features_new.csv",
    dtype={"station": "string"}
)

df = old[
    ["city", "station", "dist_subway_m"]
].merge(
    new[["city", "station", "dist_subway_m"]],
    on=["city", "station"],
    suffixes=("_old", "_new"),
    validate="one_to_one"
)

# 새 거리에도 5000m 상한을 적용해 비교
df["dist_new_capped"] = df["dist_subway_m_new"].clip(
    upper=5000
)

df["match_before"] = np.isclose(
    df["dist_subway_m_old"],
    df["dist_subway_m_new"],
    atol=0.01,
    rtol=0
)

df["match_after"] = np.isclose(
    df["dist_subway_m_old"],
    df["dist_new_capped"],
    atol=0.01,
    rtol=0
)

print("=" * 55)
print("1. 전체 비교")
print("대여소 수:", len(df))
print("상한 적용 전 일치:", df["match_before"].sum())
print("상한 적용 후 일치:", df["match_after"].sum())

print("\n" + "=" * 55)
print("2. 도시별 비교")

print(
    df.groupby("city").agg(
        stations=("station", "size"),
        match_before=("match_before", "sum"),
        match_after=("match_after", "sum")
    ).to_string()
)

print("\n" + "=" * 55)
print("3. 기존 거리 5000m인 대여소 수")

print(
    df.assign(
        old_5000=df["dist_subway_m_old"].eq(5000)
    ).groupby("city")["old_5000"].sum().to_string()
)

print("\n" + "=" * 55)
print("4. 상한 적용 후에도 다른 대여소 10개")

different = df[~df["match_after"]].copy()

different["diff"] = (
    different["dist_new_capped"]
    - different["dist_subway_m_old"]
).abs()

print(
    different.sort_values(
        "diff", ascending=False
    )[
        [
            "city",
            "station",
            "dist_subway_m_old",
            "dist_subway_m_new",
            "dist_new_capped",
            "diff"
        ]
    ].head(10).to_string(index=False)
)

print("\n지하철 거리 상한 검사 완료!")
