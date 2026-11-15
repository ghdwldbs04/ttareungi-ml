
from pathlib import Path
from collections import Counter
import unicodedata

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

def nfc(value):
    return unicodedata.normalize("NFC", str(value))

sgis_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and "SGIS" in nfc(p.name).upper()
]

if len(sgis_dirs) != 1:
    raise ValueError(
        f"SGIS 폴더 수 확인 필요: {len(sgis_dirs)}"
    )

SGIS = sgis_dirs[0]
STAT = SGIS / "1. 통계"
CODE = SGIS / "3. 코드집"

topics = [
    "인구",
    "가구",
    "주택",
    "사업체",
    "종사자"
]

print("=" * 70)
print("1. SGIS 코드집 파일 목록")

if CODE.exists():
    code_files = sorted(
        [p for p in CODE.rglob("*") if p.is_file()],
        key=lambda p: nfc(str(p))
    )

    print("코드집 파일 수:", len(code_files))

    for p in code_files:
        print(nfc(p.relative_to(BASE)))

else:
    print("코드집 폴더를 찾지 못했습니다.")

print("\n" + "=" * 70)
print("2. 통계 종류별 전체 통계항목 코드")

all_csv = list(STAT.rglob("*.csv"))

for topic in topics:
    files = sorted(
        [
            p for p in all_csv
            if f"_{topic}_" in nfc(p.name)
        ],
        key=lambda p: nfc(str(p))
    )

    print("\n" + "-" * 70)
    print(f"[{topic}]")
    print("파일 수:", len(files))

    code_counts = Counter()
    total_rows = 0
    years = set()
    non_numeric_count = 0
    non_numeric_examples = Counter()

    for path in files:
        df = pd.read_csv(
            path,
            encoding="cp949",
            dtype=str,
            low_memory=False
        )

        required = [
            "기준연도",
            "격자코드",
            "통계항목",
            "통계값"
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"필수 열 누락: {path.name}, {missing}"
            )

        total_rows += len(df)

        code_counts.update(
            df["통계항목"]
            .dropna()
            .str.strip()
            .tolist()
        )

        years.update(
            df["기준연도"]
            .dropna()
            .unique()
            .tolist()
        )

        values = df["통계값"].str.strip()

        numeric = pd.to_numeric(
            values,
            errors="coerce"
        )

        invalid = (
            values.notna()
            & numeric.isna()
        )

        non_numeric_count += int(invalid.sum())

        non_numeric_examples.update(
            values[invalid].tolist()
        )

    print("전체 행 수:", total_rows)
    print("기준연도:", sorted(years))
    print("고유 통계항목 수:", len(code_counts))
    print("숫자로 변환 불가한 통계값:", non_numeric_count)

    if non_numeric_examples:
        print(
            "숫자 변환 불가 값 예시:",
            non_numeric_examples.most_common(10)
        )

    print("\n통계항목별 행 수:")

    for code, count in sorted(code_counts.items()):
        print(f"{code}: {count}")

print("\nSGIS 통계항목 코드 검사 완료!")
