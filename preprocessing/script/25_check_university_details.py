
from pathlib import Path
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

files = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "대학교" in unicodedata.normalize("NFC", p.name)
    and "좌표" in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"대학교 파일 확인 필요: {files}")

universities = pd.read_excel(files[0])

print("원본 행 수:", len(universities))

# 1. 중복 검사
print("\n1. 중복 검사")
print(
    "완전히 동일한 행:",
    universities.duplicated().sum()
)
print(
    "학교코드변환 중복:",
    universities["학교코드변환"].duplicated().sum()
)
print(
    "학교명 중복:",
    universities["학교명"].duplicated().sum()
)
print(
    "좌표 중복:",
    universities.duplicated(["위도", "경도"]).sum()
)

# 2. 학교 유형 확인
print("\n2. 학교구분")
print(
    universities["학교구분"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n3. 학제")
print(
    universities["학제"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n4. 본교/분교")
print(
    universities["본분교"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n5. 학교상태")
print(
    universities["학교상태"]
    .value_counts(dropna=False)
    .to_string()
)

# 3. 좌표 유효성
lat = pd.to_numeric(
    universities["위도"], errors="coerce"
)
lon = pd.to_numeric(
    universities["경도"], errors="coerce"
)

invalid = (
    ~lat.between(33, 39)
    | ~lon.between(124, 132)
)

print("\n6. 좌표 검사")
print("좌표 이상 행:", invalid.sum())

if invalid.any():
    print(
        universities.loc[
            invalid,
            ["학교명", "지역", "위도", "경도"]
        ].head(10).to_string(index=False)
    )

# 4. 대상 도시의 대학 확인
print("\n7. 대상 지역별 학교 수")

for region in ["서울", "대전", "세종", "경남"]:
    subset = universities[
        universities["지역"] == region
    ]

    print(f"{region}: {len(subset)}행")

print("\n대학교 상세 검사 완료!")
