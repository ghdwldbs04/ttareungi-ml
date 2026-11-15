
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

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

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

city_regions = {
    "seoul": "서울",
    "daejeon": "대전",
    "sejong": "세종",
    "changwon": "경남"
}

def calculate_distances(station_subset, university_subset):
    university_subset = university_subset.copy()

    university_subset["위도"] = pd.to_numeric(
        university_subset["위도"], errors="coerce"
    )
    university_subset["경도"] = pd.to_numeric(
        university_subset["경도"], errors="coerce"
    )

    university_subset = university_subset[
        university_subset["위도"].between(33, 39)
        & university_subset["경도"].between(124, 132)
    ]

    sx, sy = transformer.transform(
        station_subset["lon"].to_numpy(),
        station_subset["lat"].to_numpy()
    )

    ux, uy = transformer.transform(
        university_subset["경도"].to_numpy(),
        university_subset["위도"].to_numpy()
    )

    tree = cKDTree(
        np.column_stack([ux, uy])
    )

    distances, _ = tree.query(
        np.column_stack([sx, sy]),
        k=1
    )

    return distances


print("=" * 65)
print("대학교 거리 계산 범위별 비교")

for city, region in city_regions.items():
    station_subset = stations[
        stations["city"] == city
    ].copy()

    old = station_subset["dist_univ_m"].to_numpy()

    # A. 전국 대학 전체
    national_dist = calculate_distances(
        station_subset,
        universities
    )

    # B. 해당 광역 지역 대학만
    regional_universities = universities[
        universities["지역"] == region
    ]

    regional_dist = calculate_distances(
        station_subset,
        regional_universities
    )

    national_match = np.isclose(
        national_dist,
        old,
        atol=0.01,
        rtol=0
    )

    regional_match = np.isclose(
        regional_dist,
        old,
        atol=0.01,
        rtol=0
    )

    print("\n" + "-" * 65)
    print("도시:", city)
    print("대여소 수:", len(station_subset))

    print(
        "전국 대학 기준 일치:",
        national_match.sum(),
        "/",
        len(station_subset),
        f"({national_match.mean() * 100:.2f}%)"
    )

    print(
        "해당 지역 대학 기준 일치:",
        regional_match.sum(),
        "/",
        len(station_subset),
        f"({regional_match.mean() * 100:.2f}%)"
    )

    print(
        "전국 기준 평균 절대 차이:",
        round(np.abs(national_dist - old).mean(), 2),
        "m"
    )

    print(
        "지역 기준 평균 절대 차이:",
        round(np.abs(regional_dist - old).mean(), 2),
        "m"
    )

    if city == "changwon":
        print("\n창원 주요 대여소 확인")

        for station_id in ["318", "213"]:
            mask = (
                station_subset["station"] == station_id
            ).to_numpy()

            if mask.sum() == 1:
                i = np.flatnonzero(mask)[0]

                print(
                    f"{station_id}: "
                    f"기존 {old[i]:.1f}m / "
                    f"전국 {national_dist[i]:.1f}m / "
                    f"경남 {regional_dist[i]:.1f}m"
                )

print("\n대학교 지역 범위 검사 완료!")
