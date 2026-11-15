
from pathlib import Path
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

# Mac 한글 파일명 문제 방지
park_files = [
    p for p in FOOTAGE.rglob("*.csv")
    if "전국도시공원정보" in unicodedata.normalize("NFC", p.name)
]

print("발견한 공원 원본 파일:", len(park_files))

if len(park_files) != 1:
    raise ValueError("공원 CSV 파일을 확인해 주세요.")

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

print("\n1. 공원 원본 정보")
print("파일명:", path.name)
print("전체 행 수:", len(parks))
print("열 이름:", parks.columns.tolist())

print("\n2. 주요 열 결측값")
columns = [
    "관리번호",
    "공원명",
    "공원구분",
    "위도",
    "경도",
    "공원면적"
]
print(parks[columns].isna().sum().to_string())

print("\n3. 관리번호 중복")
print(parks["관리번호"].duplicated().sum())

print("\n4. 공원 면적 통계")
area = pd.to_numeric(parks["공원면적"], errors="coerce")
print(area.describe())
print("면적이 0 이하인 공원:", (area <= 0).sum())

print("\n5. 공원구분별 개수")
print(parks["공원구분"].value_counts().head(15).to_string())

print("\n6. 공원 데이터 예시")
print(parks[columns].head(10).to_string(index=False))

print("\n공원 원본 검사 완료!")
