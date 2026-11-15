
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

# 1. 자전거도로 원본 파일 찾기
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

print("=" * 65)
print("1. 시도명 전체 목록")

print(
    df["시도명"]
    .fillna("(결측)")
    .value_counts()
    .to_string()
)

# 2. 세종 관련 기록 검색
print("\n" + "=" * 65)
print("2. 세종 관련 기록 검색")

search_cols = [
    "시도명",
    "시군구명",
    "기점도로명주소",
    "기점지번주소",
    "종점도로명주소",
    "종점지번주소",
    "제공기관명"
]

sejong_mask = pd.Series(
    False,
    index=df.index
)

for col in search_cols:
    sejong_mask |= (
        df[col]
        .fillna("")
        .astype(str)
        .str.contains("세종", regex=False)
    )

sejong = df[sejong_mask]

print("세종 관련 기록 수:", len(sejong))

if len(sejong) > 0:
    print(
        sejong[
            ["노선명", "시도명", "시군구명", "제공기관명"]
        ].head(20).to_string(index=False)
    )

# 3. 대상 도시 분류
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

target = df[df["city"] != "other"].copy()

# 4. 좌표 유효성 검사
for col in [
    "기점위도", "기점경도",
    "종점위도", "종점경도"
]:
    target[col] = pd.to_numeric(
        target[col],
        errors="coerce"
    )

start_valid = (
    target["기점위도"].between(33, 39)
    & target["기점경도"].between(124, 132)
)

end_valid = (
    target["종점위도"].between(33, 39)
    & target["종점경도"].between(124, 132)
)

target["coord_status"] = np.select(
    [
        start_valid & end_valid,
        start_valid & ~end_valid,
        ~start_valid & end_valid
    ],
    ["both", "start_only", "end_only"],
    default="neither"
)

# 5. 주소 존재 여부
address_cols = [
    "기점도로명주소",
    "기점지번주소",
    "종점도로명주소",
    "종점지번주소"
]

for col in address_cols:
    target[col] = (
        target[col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

target["start_address"] = (
    (target["기점도로명주소"] != "")
    | (target["기점지번주소"] != "")
)

target["end_address"] = (
    (target["종점도로명주소"] != "")
    | (target["종점지번주소"] != "")
)

target["any_address"] = (
    target["start_address"]
    | target["end_address"]
)

target["both_addresses"] = (
    target["start_address"]
    & target["end_address"]
)

# 6. 도시별 주소 보유 상태
print("\n" + "=" * 65)
print("3. 도시별 주소 보유 상태")

summary = target.groupby("city").agg(
    total=("city", "size"),
    start_address=("start_address", "sum"),
    end_address=("end_address", "sum"),
    any_address=("any_address", "sum"),
    both_addresses=("both_addresses", "sum")
)

print(summary.to_string())

# 7. 좌표가 모두 없는 기록의 주소 확인
print("\n" + "=" * 65)
print("4. 좌표가 모두 없는 기록의 주소 상태")

missing_coords = target[
    target["coord_status"] == "neither"
]

missing_summary = missing_coords.groupby("city").agg(
    records=("city", "size"),
    any_address=("any_address", "sum"),
    both_addresses=("both_addresses", "sum")
)

print(missing_summary.to_string())

# 8. 창원 원본 주소 예시
print("\n" + "=" * 65)
print("5. 창원 자전거도로 주소 예시")

changwon = target[
    target["city"] == "changwon"
]

print(
    changwon[
        [
            "노선명",
            "기점도로명주소",
            "기점지번주소",
            "종점도로명주소",
            "종점지번주소",
            "총길이(km)"
        ]
    ].head(10).to_string(index=False)
)

# 9. 좌표도 주소도 없는 기록
print("\n" + "=" * 65)
print("6. 좌표와 주소가 모두 없는 기록")

unusable = target[
    (target["coord_status"] == "neither")
    & (~target["any_address"])
]

print("전체:", len(unusable))
print(
    unusable["city"]
    .value_counts()
    .to_string()
)

print("\n자전거도로 주소 검사 완료!")
    