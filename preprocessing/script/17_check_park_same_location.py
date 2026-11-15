
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

# 1. 공원명 + 위도 + 경도가 같은 행
keys = ["공원명", "위도", "경도"]

same_location = parks[
    parks.duplicated(keys, keep=False)
].copy()

print("1. 공원명과 좌표가 같은 기록")
print("해당 전체 행 수:", len(same_location))
print(
    "중복으로 추가된 행 수:",
    parks.duplicated(keys).sum()
)

print("\n예시 20행")
print(
    same_location[
        ["관리번호", "공원명", "위도", "경도", "공원면적", "데이터기준일자"]
    ].head(20).to_string(index=False)
)

# 2. 같은 공원명 + 좌표에서 면적이 다른 경우
if not same_location.empty:
    area_diff = (
        same_location.groupby(keys)["공원면적"]
        .nunique()
    )

    print("\n2. 같은 공원명과 좌표인데 면적이 다른 경우")
    print("해당 그룹 수:", (area_diff > 1).sum())

# 3. 좌표만 완전히 같은 경우
coord_keys = ["위도", "경도"]

print("\n3. 좌표가 완전히 같은 기록")
print(
    "중복으로 추가된 행 수:",
    parks.duplicated(coord_keys).sum()
)

print("\n공원 위치 중복 검사 완료!")
