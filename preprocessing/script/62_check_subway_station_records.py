
from pathlib import Path
import unicodedata

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. 지하철 원본 엑셀 찾기")

candidates = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "도시철도역사정보" in nfc(p.name)
]

print("발견된 파일 수:", len(candidates))

for p in candidates:
    print(nfc(p.relative_to(BASE)))

if len(candidates) != 1:
    raise ValueError("지하철 원본 파일 확인 필요")

SOURCE = candidates[0]

print("\n" + "=" * 70)
print("2. 엑셀 시트 확인")

book = pd.ExcelFile(SOURCE)

print("시트 목록:", book.sheet_names)

for sheet in book.sheet_names:
    df = pd.read_excel(
        SOURCE,
        sheet_name=sheet,
        nrows=5
    )

    print("\n시트:", sheet)
    print("열 목록:")
    print(df.columns.tolist())
    print("앞 3행:")
    print(df.head(3).to_string(index=False))

print("\n" + "=" * 70)
print("3. 첫 번째 시트 전체 구조")

df = pd.read_excel(
    SOURCE,
    sheet_name=book.sheet_names[0]
)

print("전체 행 수:", len(df))
print("전체 열 수:", len(df.columns))

print("\n열별 결측 수:")
print(df.isna().sum().to_string())

print("\n" + "=" * 70)
print("4. 주요 역명 검색")

name_candidates = [
    c for c in df.columns
    if "역명" in nfc(c)
    or "역사명" in nfc(c)
]

print("역명 후보 열:", name_candidates)

keywords = [
    "서울역",
    "시청",
    "잠실",
    "노원",
    "회현",
    "을지로",
    "합정",
]

if name_candidates:
    name_col = name_candidates[0]

    for keyword in keywords:
        matches = df[
            df[name_col]
            .astype(str)
            .str.contains(
                keyword,
                regex=False,
                na=False
            )
        ]

        print("\n" + "-" * 60)
        print("검색어:", keyword)
        print("일치 행 수:", len(matches))

        if len(matches) > 0:
            print(
                matches.head(15).to_string(
                    index=False,
                    max_cols=12
                )
            )
else:
    print("역명 열을 자동으로 찾지 못했습니다.")

print("\n지하철 원본 역 목록 조사 완료!")
print("파일 저장 및 수정 없음")
