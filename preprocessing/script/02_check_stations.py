
from pathlib import Path
import pandas as pd

# 프로젝트 폴더 경로
BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "processed" / "stations_clean.csv"

# CSV 읽기
df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")

print("=" * 55)
print("1. 데이터 기본 정보")
print("=" * 55)
print("전체 행 수:", len(df))
print("전체 열 수:", len(df.columns))
print("변수명:", list(df.columns))

# 필수 열 확인
required = ["city", "station", "lat", "lon"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"필수 열이 없습니다: {missing}")

print("\n" + "=" * 55)
print("2. 도시별 대여소 수")
print("=" * 55)
print(df["city"].value_counts(dropna=False).to_string())

print("\n" + "=" * 55)
print("3. 대여소 ID 중복 확인")
print("=" * 55)

duplicate = df.duplicated(
    subset=["city", "station"],
    keep=False
)

print("중복된 행 수:", duplicate.sum())

if duplicate.any():
    print(df.loc[duplicate, required].head(20).to_string(index=False))

print("\n" + "=" * 55)
print("4. 필수 열 결측값")
print("=" * 55)
print(df[required].isna().sum().to_string())

print("\n" + "=" * 55)
print("5. 좌표 범위 확인")
print("=" * 55)

for col in ["lat", "lon"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df[["lat", "lon"]].describe().to_string())

# 한국의 대략적인 위도/경도 범위를 벗어나는 값 확인
invalid = (
    df["lat"].isna()
    | df["lon"].isna()
    | ~df["lat"].between(33, 39)
    | ~df["lon"].between(124, 132)
)

print("\n좌표 결측 또는 범위 이상 행 수:", invalid.sum())

if invalid.any():
    print(df.loc[invalid, required].head(20).to_string(index=False))

print("\n" + "=" * 55)
print("6. 대여소 데이터 예시")
print("=" * 55)
print(df[required].head(10).to_string(index=False))

print("\n대여소 기준 데이터 검사 완료!")
