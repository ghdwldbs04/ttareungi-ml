
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

parks = pd.read_csv(
    PROCESSED / "parks_clean.csv"
)

# 기존 피처와 차이가 발생한 서울 대여소
target_ids = ["1516", "1520", "1535", "1540"]

targets = stations[
    (stations["city"] == "seoul")
    & (stations["station"].isin(target_ids))
].copy()

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

park_x, park_y = transformer.transform(
    parks["경도"].to_numpy(),
    parks["위도"].to_numpy()
)

print("차이 발생 대여소 수:", len(targets))

for _, station in targets.iterrows():
    sx, sy = transformer.transform(
        station["lon"],
        station["lat"]
    )

    distances = np.sqrt(
        (park_x - sx) ** 2
        + (park_y - sy) ** 2
    )

    nearby = parks.loc[
        distances <= 500,
        [
            "관리번호",
            "공원명",
            "공원구분",
            "공원면적",
            "위도",
            "경도"
        ]
    ].copy()

    nearby["거리_m"] = distances[
        distances <= 500
    ]

    print("\n" + "=" * 60)
    print("서울 대여소:", station["station"])
    print("반경 500m 공원 수:", len(nearby))
    print("공원 면적 합계:", nearby["공원면적"].sum())

    print(
        nearby.sort_values("거리_m")
        .to_string(index=False)
    )

print("\n공원 차이 원인 확인 완료!")
