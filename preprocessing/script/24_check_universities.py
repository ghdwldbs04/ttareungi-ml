
from pathlib import Path
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

# 대학교 좌표정보 엑셀 찾기
files = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "대학교" in unicodedata.normalize("NFC", p.name)
    and "좌표" in unicodedata.normalize("NFC", p.name)
]

print("발견한 대학교 파일 수:", len(files))

for p in files:
    print("파일:", p.name)

if len(files) != 1:
    raise ValueError("대학교 원본 파일을 확인해 주세요.")

path = files[0]

# 엑셀 시트 구조 확인
excel = pd.ExcelFile(path)

print("\n1. 시트 이름")
print(excel.sheet_names)

for sheet in excel.sheet_names:
    df = pd.read_excel(path, sheet_name=sheet)

    print("\n" + "=" * 55)
    print("시트:", sheet)
    print("행 수:", len(df))
    print("열 이름:", df.columns.tolist())

    print("\n결측값 상위 15개 열")
    print(
        df.isna().sum()
        .sort_values(ascending=False)
        .head(15)
        .to_string()
    )

    print("\n처음 5행")
    print(df.head(5).to_string(index=False))

print("\n대학교 원본 검사 완료!")
