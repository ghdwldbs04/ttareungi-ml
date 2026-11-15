
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

# 1. 데이터 불러오기
files = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "대학교" in unicodedata.normalize("NFC", p.name)
    and "좌표" in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"대학교 원본 파일 확인 필요: {files}")

universities = pd.read_excel(files[0])

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

sx, sy = transformer.transform(
    stations["lon"].to_numpy(),
    stations["lat"].to_numpy()
)

station_xy = np.column_stack([sx, sy])

# 3. 대학 필터별 비교 함수
def compare_filter(name, subset):
    subset = subset.copy()

    subset["위도"] = pd.to_numeric(
        subset["위도"], errors="coerce"
    )
    subset["경도"] = pd.to_numeric(
        subset["경도"], errors="coerce"
    )

    subset = subset[
        subset["위도"].between(33, 39)
        & subset["경도"].between(124, 132)
    ]

    if len(subset) == 0:
        print(f"{name}: 대학 데이터 없음")
        return

    ux, uy = transformer.transform(
        subset["경도"].to_numpy(),
        subset["위도"].to_numpy()
    )

    tree = cKDTree(np.column_stack([ux, uy]))

    distances, _ = tree.query(station_xy, k=1)

    old = pd.to_numeric(
        stations["dist_univ_m"],
        errors="coerce"
    ).to_numpy()

    valid = np.isfinite(old) & np.isfinite(distances)

    differences = np.abs(distances[valid] - old[valid])

    matches = np.isclose(
        distances[valid],
        old[valid],
        atol=0.01,
        rtol=0
    )

    print(f"\n[{name}]")
    print("대학 기록 수:", len(subset))
    print("기존 거리 일치:", matches.sum(), "/", valid.sum())
    print("일치율:", round(matches.mean() * 100, 2), "%")
    print("평균 절대 차이:", round(differences.mean(), 2), "m")

    # 차이가 컸던 대표 대여소
    for city, station_id in [
        ("sejong", "SJ-00083"),
        ("changwon", "318"),
    ]:
        mask = (
            (stations["city"] == city)
            & (stations["station"] == station_id)
        )

        indices = np.flatnonzero(mask.to_numpy())

        if len(indices) == 1:
            i = indices[0]
            print(
                f"{city} {station_id}: "
                f"기존 {old[i]:.1f}m / "
                f"현재 {distances[i]:.1f}m"
            )


# 4. 비교할 필터 정의
regular_types = ["대학", "전문대학", "대학원대학"]

active = universities[
    universities["학교상태"].isin(["기존", "신설"])
]

regular = universities[
    universities["학교구분"].isin(regular_types)
]

regular_active = universities[
    universities["학교구분"].isin(regular_types)
    & universities["학교상태"].isin(["기존", "신설"])
]

regular_active_no_cyber = regular_active[
    ~regular_active["학제"].astype(str).str.contains(
        "사이버", na=False
    )
]

print("=" * 60)
print("대학교 필터별 기존 거리 일치율 비교")

compare_filter(
    "A. 원본 전체",
    universities
)

compare_filter(
    "B. 기존·신설 전체",
    active
)

compare_filter(
    "C. 대학·전문대학·대학원대학 전체",
    regular
)

compare_filter(
    "D. 대학 유형 + 기존·신설",
    regular_active
)

compare_filter(
    "E. 대학 유형 + 기존·신설 + 사이버 제외",
    regular_active_no_cyber
)

print("\n대학교 필터 영향 검사 완료!")
