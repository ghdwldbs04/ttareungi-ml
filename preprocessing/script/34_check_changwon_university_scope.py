
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

# 1. 대학 원본 파일 찾기
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

changwon = stations[
    stations["city"] == "changwon"
].copy().reset_index(drop=True)

# 2. 창원시 소재 대학 구분
# 주소의 공백 차이를 고려하여 검사
address = (
    universities["도로명 주소"]
    .fillna("")
    .astype(str)
    .str.replace(" ", "", regex=False)
)

is_changwon = address.str.contains(
    "창원시",
    regex=False
)

national = universities.copy()

gyeongnam = universities[
    universities["지역"] == "경남"
].copy()

changwon_only = universities[
    is_changwon
].copy()

# 3. 거리 계산
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

sx, sy = transformer.transform(
    changwon["lon"].to_numpy(),
    changwon["lat"].to_numpy()
)

station_xy = np.column_stack([sx, sy])

def calculate_distances(data):
    data = data.copy()

    data["위도"] = pd.to_numeric(
        data["위도"], errors="coerce"
    )
    data["경도"] = pd.to_numeric(
        data["경도"], errors="coerce"
    )

    data = data[
        data["위도"].between(33, 39)
        & data["경도"].between(124, 132)
    ]

    if len(data) == 0:
        raise ValueError("대학 데이터가 없습니다.")

    ux, uy = transformer.transform(
        data["경도"].to_numpy(),
        data["위도"].to_numpy()
    )

    tree = cKDTree(
        np.column_stack([ux, uy])
    )

    distances, _ = tree.query(
        station_xy,
        k=1
    )

    return distances

old = pd.to_numeric(
    changwon["dist_univ_m"],
    errors="coerce"
).to_numpy()

dist_national = calculate_distances(national)
dist_gyeongnam = calculate_distances(gyeongnam)
dist_changwon = calculate_distances(changwon_only)

# 4. 전체 일치율 비교
print("=" * 65)
print("창원 대학 거리 계산 범위 비교")
print("창원 대여소 수:", len(changwon))
print("전국 대학 기록 수:", len(national))
print("경남 대학 기록 수:", len(gyeongnam))
print("창원시 대학 기록 수:", len(changwon_only))

for label, distances in [
    ("전국", dist_national),
    ("경남", dist_gyeongnam),
    ("창원시", dist_changwon),
]:
    matches = np.isclose(
        old,
        distances,
        atol=0.01,
        rtol=0
    )

    print(f"\n[{label}]")
    print("일치:", matches.sum(), "/", len(changwon))
    print("일치율:", round(matches.mean() * 100, 2), "%")
    print(
        "평균 절대 차이:",
        round(np.abs(old - distances).mean(), 2),
        "m"
    )

# 5. 전국 기준과 차이 나는 대여소 7개 확인
national_matches = np.isclose(
    old,
    dist_national,
    atol=0.01,
    rtol=0
)

result = changwon.loc[
    ~national_matches,
    ["city", "station"]
].copy()

result["기존거리_m"] = old[~national_matches]
result["전국거리_m"] = dist_national[~national_matches]
result["경남거리_m"] = dist_gyeongnam[~national_matches]
result["창원시거리_m"] = dist_changwon[~national_matches]

result["전국차이_m"] = (
    result["전국거리_m"] - result["기존거리_m"]
).abs()

result = result.sort_values(
    "전국차이_m",
    ascending=False
)

print("\n" + "=" * 65)
print("전국 기준과 불일치하는 창원 대여소")
print(result.to_string(index=False))

print("\n창원 대학 지역 범위 검사 완료!")
