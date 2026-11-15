
from pathlib import Path
import unicodedata

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

# 1. 자전거도로 원본 파일 찾기
files = [
    p for p in FOOTAGE.rglob("*.csv")
    if "전국자전거도로표준데이터"
    in unicodedata.normalize("NFC", p.name)
]

print("=" * 60)
print("1. 자전거도로 파일 확인")
print("발견한 파일 수:", len(files))

for p in files:
    print("파일:", p)

if len(files) != 1:
    raise ValueError("자전거도로 원본 CSV 파일을 확인해 주세요.")

file_path = files[0]

# 2. 인코딩 확인 후 불러오기
df = None

for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            low_memory=False
        )
        print("\n사용한 인코딩:", encoding)
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise ValueError("CSV 인코딩을 확인할 수 없습니다.")

# 3. 기본 구조
print("\n" + "=" * 60)
print("2. 원본 데이터 구조")

print("전체 행 수:", len(df))
print("전체 열 수:", len(df.columns))

print("\n전체 열 이름:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")

# 4. 결측값 확인
print("\n" + "=" * 60)
print("3. 열별 결측값")

missing = pd.DataFrame({
    "결측 수": df.isna().sum(),
    "결측 비율(%)": (
        df.isna().mean() * 100
    ).round(2)
})

print(
    missing.sort_values(
        "결측 수",
        ascending=False
    ).to_string()
)

# 5. 좌표 및 도로 관련 열 찾기
print("\n" + "=" * 60)
print("4. 좌표·도로 관련 열")

keywords = [
    "위도", "경도", "좌표",
    "시점", "종점", "노선",
    "도로", "길이", "연장",
    "geometry", "wkt"
]

related_cols = [
    col for col in df.columns
    if any(
        keyword.lower() in col.lower()
        for keyword in keywords
    )
]

print("관련 열:", related_cols)

# 6. 데이터 예시
print("\n" + "=" * 60)
print("5. 처음 5개 행")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

print(df.head(5).to_string(index=False))

# 7. 중복 확인
print("\n" + "=" * 60)
print("6. 완전히 동일한 행")

print("전체 행 중복 수:", df.duplicated().sum())

print("\n자전거도로 원본 구조 검사 완료!")
