
from pathlib import Path
from collections import Counter
import unicodedata

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

def nfc(value):
    return unicodedata.normalize("NFC", str(value))

# SGIS 폴더 찾기
sgis_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and (
        "SGIS" in nfc(p.name).upper()
        or "격자" in nfc(p.name)
    )
]

print("=" * 70)
print("1. SGIS 폴더 확인")

print("발견된 폴더 수:", len(sgis_dirs))

for folder in sgis_dirs:
    print("-", nfc(folder.name))

if not sgis_dirs:
    raise FileNotFoundError(
        "SGIS 격자 데이터 폴더를 찾지 못했습니다."
    )

# 전체 파일 수집
all_files = []

for folder in sgis_dirs:
    all_files.extend(
        p for p in folder.rglob("*")
        if p.is_file()
    )

print("\n" + "=" * 70)
print("2. 확장자별 파일 수")

ext_counts = Counter(
    p.suffix.lower()
    for p in all_files
)

print("전체 파일 수:", len(all_files))

for ext, count in sorted(ext_counts.items()):
    print(f"{ext or '(확장자 없음)'}: {count}")

print("\n" + "=" * 70)
print("3. 최상위 폴더 구성")

for folder in sgis_dirs:
    print(f"\n[{nfc(folder.name)}]")

    for child in sorted(
        folder.iterdir(),
        key=lambda p: nfc(p.name)
    ):
        kind = "폴더" if child.is_dir() else "파일"
        print(f"{kind}: {nfc(child.name)}")

print("\n" + "=" * 70)
print("4. 통계 관련 파일 목록")

data_exts = {
    ".csv",
    ".xlsx",
    ".xls",
    ".txt",
    ".dbf",
    ".shp"
}

data_files = [
    p for p in all_files
    if p.suffix.lower() in data_exts
]

print("통계·공간 데이터 파일 수:", len(data_files))

for p in sorted(
    data_files,
    key=lambda p: nfc(str(p))
)[:100]:
    print(nfc(p.relative_to(BASE)))

if len(data_files) > 100:
    print(f"... 외 {len(data_files) - 100}개")

print("\n" + "=" * 70)
print("5. 파일명에서 통계 주제 검색")

topics = [
    "인구",
    "사업체",
    "종사자",
    "가구",
    "주택",
    "population",
    "business",
    "employee",
    "grid"
]

for topic in topics:
    matches = [
        p for p in all_files
        if topic.lower() in nfc(p.name).lower()
    ]

    print(f"{topic}: {len(matches)}개")

print("\nSGIS 격자 파일 구조 검사 완료!")
