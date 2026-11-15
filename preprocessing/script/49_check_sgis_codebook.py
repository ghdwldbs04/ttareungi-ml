
from pathlib import Path
import unicodedata

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

def nfc(value):
    return unicodedata.normalize("NFC", str(value))

# 1. SGIS 코드집 찾기
files = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "statistics_code" in nfc(p.name).lower()
]

if len(files) != 1:
    raise ValueError(
        f"SGIS 통계 코드집 파일 확인 필요: {files}"
    )

path = files[0]

print("=" * 70)
print("1. SGIS 통계 코드집 파일")

print("파일:", nfc(path.relative_to(BASE)))

# 2. 시트 이름
book = pd.ExcelFile(path)

print("\n시트 목록:")
for name in book.sheet_names:
    print("-", name)

# 3. 시트별 구조 확인
print("\n" + "=" * 70)
print("2. 시트별 열 구조와 앞부분")

for sheet in book.sheet_names:
    print("\n" + "-" * 70)
    print(f"[시트: {sheet}]")

    df = pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        nrows=30
    )

    print("읽은 행 수:", len(df))
    print("열 수:", len(df.columns))

    print("\n앞 20행:")
    print(
        df.head(20)
        .fillna("")
        .to_string(
            index=False,
            header=False
        )
    )

# 4. 주요 코드가 어느 시트에 있는지 검색
print("\n" + "=" * 70)
print("3. 주요 통계 코드 검색")

target_codes = [
    "to_in_001",
    "to_in_007",
    "to_in_008",
    "to_ga_001",
    "to_ho_001",
    "to_fa_010",
    "to_em_020",
    "in_age_006",
    "cp_bnu_009",
    "cp_bem_009"
]

for sheet in book.sheet_names:
    df = pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        dtype=str
    )

    # 모든 셀을 문자열로 변환
    text_df = df.fillna("").astype(str)

    for code in target_codes:
        matches = text_df.apply(
            lambda col: col.str.contains(
                code,
                regex=False,
                case=False
            )
        ).any(axis=1)

        found = df.loc[matches]

        if not found.empty:
            print(f"\n[시트: {sheet} / 코드: {code}]")
            print(
                found.head(5)
                .fillna("")
                .to_string(
                    index=False,
                    header=False
                )
            )

print("\nSGIS 공식 코드집 검사 완료!")
