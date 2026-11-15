
from pathlib import Path
from collections import Counter
import unicodedata

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"


def nfc(value):
    return unicodedata.normalize("NFC", str(value))


print("=" * 70)
print("1. 경사도 데이터 폴더 찾기")

slope_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and (
        "Slope" in nfc(p.name)
        or "경사" in nfc(p.name)
    )
]

if len(slope_dirs) != 1:
    raise ValueError(
        f"경사도 폴더 확인 필요: {len(slope_dirs)}"
    )

SLOPE = slope_dirs[0]

print("폴더:", nfc(SLOPE.relative_to(BASE)))

print("\n" + "=" * 70)
print("2. 최상위 폴더 및 파일 목록")

for p in sorted(
    SLOPE.iterdir(),
    key=lambda x: nfc(x.name)
):
    kind = "폴더" if p.is_dir() else "파일"
    print(f"[{kind}] {nfc(p.name)}")

print("\n" + "=" * 70)
print("3. 전체 파일 확장자별 개수")

all_files = [
    p for p in SLOPE.rglob("*")
    if p.is_file()
]

ext_counts = Counter(
    nfc(p.suffix).lower() or "(확장자 없음)"
    for p in all_files
)

print("전체 파일 수:", len(all_files))

for ext, count in sorted(ext_counts.items()):
    print(f"{ext}: {count}개")

print("\n" + "=" * 70)
print("4. 하위 폴더별 파일 개수")

folder_counts = Counter()

for p in all_files:
    relative = p.relative_to(SLOPE)

    if len(relative.parts) > 1:
        group = nfc(relative.parts[0])
    else:
        group = "(최상위)"

    folder_counts[group] += 1

for group, count in sorted(folder_counts.items()):
    print(f"{group}: {count}개")

print("\n" + "=" * 70)
print("5. 주요 공간정보 파일 예시")

interesting_exts = {
    ".tif",
    ".tiff",
    ".img",
    ".asc",
    ".shp",
    ".gpkg",
    ".vrt",
    ".prj",
    ".xml",
}

interesting = sorted(
    [
        p for p in all_files
        if nfc(p.suffix).lower() in interesting_exts
    ],
    key=lambda p: nfc(str(p))
)

print("주요 공간정보 파일 수:", len(interesting))

for p in interesting[:50]:
    print(nfc(p.relative_to(BASE)))

if len(interesting) > 50:
    print(
        f"... 나머지 {len(interesting) - 50}개 생략"
    )

print("\n" + "=" * 70)
print("6. 용량이 큰 파일 상위 15개")

largest = sorted(
    all_files,
    key=lambda p: p.stat().st_size,
    reverse=True
)[:15]

for p in largest:
    size_mb = p.stat().st_size / (1024 ** 2)

    print(
        f"{size_mb:10.2f} MB | "
        f"{nfc(p.relative_to(BASE))}"
    )

print("\n경사도 원본 파일 조사 완료!")
