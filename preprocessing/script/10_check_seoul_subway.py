
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
    ["city", "station", "lat", "lon", "dist_subway_m", "subway_1000"]
].merge(
    new[
        ["city", "station", "dist_subway_m", "subway_1000"]
    ],
    on=["city", "station"],
    suffixes=("_old", "_new"),
    validate="one_to_one"
)

seoul = df[df["city"] == "seoul"].copy()

seoul["distance_diff"] = (
    seoul["dist_subway_m_new"]
    - seoul["dist_subway_m_old"]
)

seoul["distance_match"] = np.isclose(
    seoul["dist_subway_m_old"],
    seoul["dist_subway_m_new"],
    atol=0.01,
    rtol=0
)

different = seoul[~seoul["distance_match"]].copy()

print("1. 서울 지하철 거리 비교")
print("서울 전체 대여소:", len(seoul))
print("거리 차이 발생:", len(different))

print("\n2. 차이의 방향")
print(
    "새 거리가 더 짧은 대여소:",
    (different["distance_diff"] < 0).sum()
)
print(
    "새 거리가 더 긴 대여소:",
    (different["distance_diff"] > 0).sum()
)

print("\n3. 차이 규모")
print(
    different["distance_diff"].abs().describe()
)

print("\n4. 거리 차이가 큰 대여소 15개")
print(
    different.assign(
        abs_diff=different["distance_diff"].abs()
    ).sort_values(
        "abs_diff", ascending=False
    )[
        [
            "station", "lat", "lon",
            "dist_subway_m_old",
            "dist_subway_m_new",
            "subway_1000_old",
            "subway_1000_new",
            "distance_diff"
        ]
    ].head(15).to_string(index=False)
)

print("\n서울 지하철 차이 검사 완료!")
