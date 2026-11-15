
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

files = [
    p for p in FOOTAGE.rglob("*.csv")
    if "전국자전거도로표준데이터"
    in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"원본 파일 확인 필요: {files}")

df = pd.read_csv(
    files[0],
    encoding="cp949",
    low_memory=False
)

# 1. 대상 도시 구분
sido = df["시도명"].fillna("").astype(str)
sigungu = df["시군구명"].fillna("").astype(str)

conditions = [
    sido.str.contains("서울", regex=False),
    sido.str.contains("대전", regex=False),
    sido.str.contains("세종", regex=False),
    sigungu.str.contains("창원", regex=False)
]

df["city"] = np.select(
    conditions,
    ["seoul", "daejeon", "sejong", "changwon"],
    default="other"
)

target = df[
    df["city"] != "other"
].copy()

# 2. 좌표 및 길이 숫자 변환
coord_cols = [
    "기점위도", "기점경도",
    "종점위도", "종점경도"
]

for col in coord_cols:
    target[col] = pd.to_numeric(
        target[col],
        errors="coerce"
    )

target["length_m"] = pd.to_numeric(
    target["총길이(km)"],
    errors="coerce"
) * 1000

start_valid = (
    target["기점위도"].between(33, 39)
    & target["기점경도"].between(124, 132)
)

end_valid = (
    target["종점위도"].between(33, 39)
    & target["종점경도"].between(124, 132)
)

both = target[
    start_valid & end_valid
].copy()

# 3. 기점·종점 직선거리 계산
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

sx, sy = transformer.transform(
    both["기점경도"].to_numpy(),
    both["기점위도"].to_numpy()
)

ex, ey = transformer.transform(
    both["종점경도"].to_numpy(),
    both["종점위도"].to_numpy()
)

both["straight_m"] = np.sqrt(
    (ex - sx) ** 2 + (ey - sy) ** 2
)

both["length_ratio"] = (
    both["length_m"]
    / both["straight_m"].replace(0, np.nan)
)

print("=" * 65)
print("1. 기점·종점 좌표가 모두 있는 기록")

print("전체:", len(both))
print(both["city"].value_counts().to_string())

print("\n" + "=" * 65)
print("2. 실제 기록 길이와 직선거리 비교")

print(
    both.groupby("city")[
        ["length_m", "straight_m", "length_ratio"]
    ]
    .agg(["count", "mean", "median", "max"])
    .round(2)
    .to_string()
)

print("\n" + "=" * 65)
print("3. 길이 관련 이상 후보")

print(
    "기점·종점 직선거리 0m:",
    (both["straight_m"] < 0.01).sum()
)

print(
    "기록 길이가 직선거리보다 짧음:",
    (both["length_m"] + 1 < both["straight_m"]).sum()
)

print(
    "기록 길이가 직선거리의 3배 초과:",
    (both["length_ratio"] > 3).sum()
)

print(
    "기록 길이가 직선거리의 10배 초과:",
    (both["length_ratio"] > 10).sum()
)

print("\n길이 비율이 큰 기록 10개:")
print(
    both[
        [
            "city", "노선명",
            "length_m", "straight_m", "length_ratio"
        ]
    ]
    .sort_values("length_ratio", ascending=False)
    .head(10)
    .round(2)
    .to_string(index=False)
)

# 4. 주소 정규화
def normalize_address(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
    )

start_road = normalize_address(
    target["기점도로명주소"]
)
start_lot = normalize_address(
    target["기점지번주소"]
)
end_road = normalize_address(
    target["종점도로명주소"]
)
end_lot = normalize_address(
    target["종점지번주소"]
)

# 도로명주소 우선, 없으면 지번주소
target["start_address"] = start_road.where(
    start_road != "", start_lot
)

target["end_address"] = end_road.where(
    end_road != "", end_lot
)

# 5. 기점과 종점 순서에 상관없는 주소 쌍 생성
target["address_pair"] = [
    tuple(sorted([start, end]))
    for start, end in zip(
        target["start_address"],
        target["end_address"]
    )
]

# 같은 도시 + 같은 주소 쌍을 중복 후보로 검사
duplicate_mask = target.duplicated(
    ["city", "address_pair"],
    keep=False
)

duplicates = target[
    duplicate_mask
].copy()

print("\n" + "=" * 65)
print("4. 주소 쌍 기준 중복 후보")

print("중복 후보 기록 수:", len(duplicates))

print("\n도시별 중복 후보:")
print(
    duplicates["city"]
    .value_counts()
    .to_string()
)

print("\n창원 중복 후보 예시:")
changwon_duplicates = duplicates[
    duplicates["city"] == "changwon"
]

print(
    changwon_duplicates[
        [
            "노선명",
            "start_address",
            "end_address",
            "총길이(km)",
            "자전거도로종류"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n자전거도로 형상 및 중복 후보 검사 완료!")
    