
from pathlib import Path
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

if not STAT.exists():
    raise FileNotFoundError(STAT)

topics = [
    "인구",
    "가구",
    "주택",
    "사업체",
    "종사자"
]

# CSV 인코딩을 순서대로 시도
def read_csv_safe(path):
    errors = []

    for encoding in [
        "utf-8-sig",
        "cp949",
        "euc-kr"
    ]:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                nrows=1000
            )

            return df, encoding

        except (UnicodeError, pd.errors.ParserError) as e:
            errors.append(
                f"{encoding}: {type(e).__name__}"
            )

    raise ValueError(
        f"CSV 읽기 실패: {path}\n{errors}"
    )

print("=" * 70)
print("1. 통계 종류별 CSV 파일 수")

all_csv = list(STAT.rglob("*.csv"))

for topic in topics:
    files = [
        p for p in all_csv
        if f"_{topic}_" in nfc(p.name)
    ]

    print(f"{topic}: {len(files)}개")

print("\n" + "=" * 70)
print("2. 통계 종류별 대표 CSV 구조")

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

    if not files:
        print("해당 CSV 없음")
        continue

    path = files[0]

    print("파일:", nfc(path.relative_to(BASE)))

    df, encoding = read_csv_safe(path)

    print("인코딩:", encoding)
    print("읽은 행 수 (최대 1000행):", len(df))
    print("열 수:", len(df.columns))
    print("열 이름:")
    print(list(df.columns))

    print("\n앞 3행:")
    print(
        df.head(3).to_string(index=False)
    )

    print("\n열별 자료형:")
    print(
        df.dtypes.astype(str).to_string()
    )

    print("\n열별 결측 수 (읽은 행 기준):")
    print(
        df.isna().sum().to_string()
    )

print("\n" + "=" * 70)
print("3. 격자 경계 SHP 파일명 확인")

shp_files = sorted(
    SGIS.rglob("*.shp"),
    key=lambda p: nfc(str(p))
)

print("SHP 파일 수:", len(shp_files))

for p in shp_files[:10]:
    print(nfc(p.relative_to(BASE)))

print("\nSGIS CSV 구조 검사 완료!")
