
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "footage"

# 엑셀 파일 목록 확인
excel_files = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() == ".xlsx"
    and not p.name.startswith("~$")
]

print("발견한 엑셀 파일 수:", len(excel_files))

for p in excel_files:
    print("-", p.name)

# 지하철 관련 열을 포함한 엑셀 파일 찾기
matches = []

for path in excel_files:
    try:
        sample = pd.read_excel(path, nrows=5)
        columns = list(sample.columns)

        if "역위도" in columns and "역경도" in columns:
            matches.append(path)
    except Exception as e:
        print("읽기 실패:", path.name, str(e))

print("\n지하철 원본 후보:", len(matches))

for p in matches:
    print("-", p.name)

if len(matches) != 1:
    raise ValueError(
        "지하철 원본 파일이 정확히 1개인지 확인해 주세요."
    )

file_path = matches[0]
df = pd.read_excel(file_path)

print("\n1. 전체 행 수:", len(df))
print("2. 변수명:", list(df.columns))

print("\n3. 좌표 결측값")
print(df[["역위도", "역경도"]].isna().sum().to_string())

df["역위도"] = pd.to_numeric(df["역위도"], errors="coerce")
df["역경도"] = pd.to_numeric(df["역경도"], errors="coerce")

invalid = (
    df["역위도"].isna()
    | df["역경도"].isna()
    | ~df["역위도"].between(33, 39)
    | ~df["역경도"].between(124, 132)
)

print("\n4. 좌표 결측 또는 범위 이상:", invalid.sum())

print("\n5. 좌표 범위")
print(df[["역위도", "역경도"]].describe().to_string())

print("\n6. 앞의 10개 행")
print(df.head(10).to_string(index=False))

print("\n지하철역 원본 검사 완료!")
