
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

# 1. 원본 대학교 파일 찾기
files = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "대학교" in unicodedata.normalize("NFC", p.name)
    and "좌표" in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"대학교 원본 파일 확인 필요: {files}")

original = pd.read_excel(files[0])

cleaned = pd.read_csv(
    PROCESSED / "universities_clean.csv"
)

stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

# 2. 좌표 변환
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

# 3. 가장 가까운 학교 검색 함수
def find_nearest(station, universities, label):
    universities = universities.copy()

    universities["위도"] = pd.to_numeric(
        universities["위도"], errors="coerce"
    )
    universities["경도"] = pd.to_numeric(
        universities["경도"], errors="coerce"
    )

    universities = universities[
        universities["위도"].between(33, 39)
        & universities["경도"].between(124, 132)
    ].reset_index(drop=True)

    ux, uy = transformer.transform(
        universities["경도"].to_numpy(),
        universities["위도"].to_numpy()
    )

    sx, sy = transformer.transform(
        station["lon"],
        station["lat"]
    )

    tree = cKDTree(np.column_stack([ux, uy]))

    distances, indices = tree.query(
        [sx, sy],
        k=min(5, len(universities))
    )

    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)

    result = universities.iloc[indices][
        [
            "학교명",
            "학교구분",
            "학제",
            "학교상태",
            "지역",
            "도로명 주소"
        ]
    ].copy()

    result.insert(0, "거리_m", distances)

    print(f"\n[{label}]")
    print(result.to_string(index=False))


# 4. 차이가 큰 대여소 확인
targets = [
    ("changwon", "318"),
    ("changwon", "213"),
    ("sejong", "SJ-00083"),
]

for city, station_id in targets:
    selected = stations[
        (stations["city"] == city)
        & (stations["station"] == station_id)
    ]

    if len(selected) != 1:
        print(f"대여소 확인 필요: {city}, {station_id}")
        continue

    station = selected.iloc[0]

    print("\n" + "=" * 65)
    print("대여소:", city, station_id)
    print("기존 대학 거리:", station["dist_univ_m"], "m")

    find_nearest(
        station,
        original,
        "원본 전체 대학 기준"
    )

    find_nearest(
        station,
        cleaned,
        "새로 정제한 대학 기준"
    )

print("\n대학교 최근접 학교 검사 완료!")
