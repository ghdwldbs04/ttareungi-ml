
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

# 같은 공원명 + 좌표인데 면적이 다른 그룹
keys = ["공원명", "위도", "경도"]

area_counts = (
    parks.groupby(keys)["공원면적"]
    .nunique()
)

conflict_keys = area_counts[
    area_counts > 1
].index

conflicts = (
    parks.set_index(keys)
    .loc[conflict_keys]
    .reset_index()
)

print("1. 면적이 다른 공원 그룹 수:", len(conflict_keys))
print("해당 전체 행 수:", len(conflicts))

print("\n2. 면적이 다른 공원 상세 정보")

columns = [
    "관리번호",
    "공원명",
    "공원구분",
    "위도",
    "경도",
    "공원면적",
    "데이터기준일자",
    "제공기관명"
]

print(
    conflicts[columns]
    .sort_values(["공원명", "위도", "경도", "데이터기준일자"])
    .to_string(index=False)
)

print("\n면적 충돌 검사 완료!")
