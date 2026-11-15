
from pathlib import Path
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

park_files = [
    p for p in FOOTAGE.rglob("*.csv")
    if "전국도시공원정보" in unicodedata.normalize("NFC", p.name)
]

if len(park_files) != 1:
    raise ValueError(f"공원 원본 파일 확인 필요: {park_files}")

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

print("원본 공원 수:", len(parks))

# 1. 좌표와 면적을 숫자로 변환
for col in ["위도", "경도", "공원면적"]:
    parks[col] = pd.to_numeric(
        parks[col],
        errors="coerce"
    )

# 2. 좌표가 없거나 대한민국 범위를 벗어난 기록 제외
valid_coord = (
    parks["위도"].between(33, 39)
    & parks["경도"].between(124, 132)
)

print("좌표 이상 행 수:", (~valid_coord).sum())

parks = parks.loc[valid_coord].copy()

# 3. 면적 결측이나 음수 확인
print("면적 결측 수:", parks["공원면적"].isna().sum())
print("면적 음수 수:", (parks["공원면적"] < 0).sum())
print("면적 0인 공원 수:", (parks["공원면적"] == 0).sum())

# 4. 주요 정보가 모두 같은 경우에만 중복 제거
dedup_keys = [
    "공원명",
    "공원구분",
    "위도",
    "경도",
    "공원면적"
]

before = len(parks)

parks = parks.drop_duplicates(
    subset=dedup_keys,
    keep="first"
).copy()

print("\n중복 제거 행 수:", before - len(parks))
print("중복 제거 후 공원 수:", len(parks))

# 5. 저장
output = PROCESSED / "parks_clean.csv"

parks.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:", output)
print("\n공원 전처리 완료!")
