
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

keywords = [
    "bike_pts_500",
    "bike_km_500",
    "cycleway_m_500",
    "dist_cycleway_m",
    "cycleway",
    "자전거도로"
]

# 검색할 텍스트 파일 형식
extensions = {
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".sql",
    ".yaml",
    ".yml"
}

# 원본 CSV 등 대용량 데이터는 제외
skip_dirs = {
    "footage",
    "processed",
    ".git",
    ".venv",
    "venv",
    "__pycache__"
}

print("=" * 70)
print("1. 프로젝트 코드에서 자전거도로 피처 검색")

matches = []

for path in BASE.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() not in extensions:
        continue

    relative = path.relative_to(BASE)

    if any(part in skip_dirs for part in relative.parts):
        continue

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace"
        )
    except OSError:
        continue

    for line_no, line in enumerate(
        text.splitlines(),
        start=1
    ):
        found = [
            keyword
            for keyword in keywords
            if keyword.lower() in line.lower()
        ]

        if found:
            matches.append(
                (str(relative), line_no, found, line.strip())
            )

print("검색된 코드 줄 수:", len(matches))

print("\n일치 결과 (최대 100줄):")

for filename, line_no, found, line in matches[:100]:
    print(f"\n파일: {filename}")
    print(f"줄 번호: {line_no}")
    print(f"키워드: {', '.join(found)}")
    print(f"내용: {line[:250]}")

if len(matches) > 100:
    print(f"\n... 나머지 {len(matches) - 100}줄 생략")

print("\n" + "=" * 70)
print("2. 피처별 등장 파일")

for keyword in keywords[:4]:
    files = sorted({
        filename
        for filename, _, found, _ in matches
        if keyword in found
    })

    print(f"\n[{keyword}]")
    print("등장 파일 수:", len(files))

    for filename in files[:30]:
        print(" -", filename)

print("\n자전거도로 피처 생성 코드 검색 완료!")
