
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

path = park_files[0]

try:
    original = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )
except UnicodeDecodeError:
    original = pd.read_csv(
        path,
        encoding="cp949",
        low_memory=False
    )

cleaned = pd.read_csv(
    PROCESSED / "parks_clean.csv"
)

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

print("1. 원본 데이터의 소귀 어린이공원")

original_sogwi = original[
    original["공원명"] == "소귀"
]

print("원본 행 수:", len(original_sogwi))
print(original_sogwi[columns].to_string(index=False))

print("\n2. 전처리 후 소귀 어린이공원")

cleaned_sogwi = cleaned[
    cleaned["공원명"] == "소귀"
]

print("전처리 후 행 수:", len(cleaned_sogwi))
print(cleaned_sogwi[columns].to_string(index=False))

print("\n확인 완료!")
