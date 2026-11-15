
from pathlib import Path
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

park_files = [
    p for p in FOOTAGE.rglob("*.csv")
    if "전국도시공원정보" in unicodedata.normalize("NFC", p.name)
]

path = park_files[0]

try:
    parks = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )
except UnicodeDecodeError:
    parks = pd.read_csv(
        path,
        encoding="cp949",
        low_memory=False
    )

# 1. 완전히 동일한 행 확인
print("1. 완전히 동일한 행")
print("완전 중복 행 수:", parks.duplicated().sum())

# 2. 관리번호 중복 확인
duplicate_parks = parks[
    parks.duplicated("관리번호", keep=False)
].copy()

print("\n2. 관리번호 중복")
print("중복 관리번호 종류:", duplicate_parks["관리번호"].nunique())
print("중복 관리번호에 해당하는 전체 행:", len(duplicate_parks))

# 3. 관리번호가 같지만 다른 정보가 있는지 확인
grouped = duplicate_parks.groupby("관리번호").agg(
    공원명종류=("공원명", "nunique"),
    위도종류=("위도", "nunique"),
    경도종류=("경도", "nunique"),
    면적종류=("공원면적", "nunique")
)

different = grouped[
    (grouped["공원명종류"] > 1)
    | (grouped["위도종류"] > 1)
    | (grouped["경도종류"] > 1)
    | (grouped["면적종류"] > 1)
]

print("\n3. 관리번호가 같지만 정보가 다른 경우")
print("해당 관리번호 수:", len(different))

print("\n정보가 다른 관리번호 예시 5개")
print(
    different.head(5).to_string()
)

if not different.empty:
    sample_ids = different.head(5).index

    print("\n실제 데이터 예시")
    print(
        duplicate_parks[
            duplicate_parks["관리번호"].isin(sample_ids)
        ][
            ["관리번호", "공원명", "위도", "경도", "공원면적"]
        ].sort_values("관리번호").to_string(index=False)
    )

# 4. 면적 0 이하 확인
area = pd.to_numeric(
    parks["공원면적"],
    errors="coerce"
)

invalid_area = parks[area <= 0]

print("\n4. 면적 0 이하인 공원")
print("개수:", len(invalid_area))

print(
    invalid_area[
        ["관리번호", "공원명", "공원구분", "공원면적", "위도", "경도"]
    ].to_string(index=False)
)

print("\n공원 중복 및 면적 검사 완료!")
