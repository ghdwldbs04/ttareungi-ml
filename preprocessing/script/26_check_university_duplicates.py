
from pathlib import Path
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

files = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "대학교" in unicodedata.normalize("NFC", p.name)
    and "좌표" in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"대학교 파일 확인 필요: {files}")

df = pd.read_excel(files[0])

print("=" * 60)
print("1. 학교구분별 학교상태")

print(
    pd.crosstab(
        df["학교구분"],
        df["학교상태"]
    ).to_string()
)

print("\n" + "=" * 60)
print("2. 학교구분별 좌표 중복")

for category, group in df.groupby("학교구분"):
    duplicate_count = group.duplicated(
        ["위도", "경도"]
    ).sum()

    print(
        f"{category}: 전체 {len(group)}행, "
        f"같은 유형 내 좌표 중복 {duplicate_count}행"
    )

print("\n" + "=" * 60)
print("3. 같은 좌표에 등록된 학교 예시")

counts = (
    df.groupby(["위도", "경도"])
    .size()
    .sort_values(ascending=False)
)

for (lat, lon), count in counts.head(5).items():
    print(f"\n좌표: {lat}, {lon} / 등록 {count}행")

    subset = df[
        (df["위도"] == lat)
        & (df["경도"] == lon)
    ]

    print(
        subset[
            ["학교명", "학교구분", "학제", "본분교", "학교상태"]
        ].head(15).to_string(index=False)
    )

print("\n" + "=" * 60)
print("4. 대상 지역의 기존·신설 대학 및 전문대학")

target = df[
    df["지역"].isin(["서울", "대전", "세종", "경남"])
    & df["학교구분"].isin(["대학", "전문대학", "대학원대학"])
    & df["학교상태"].isin(["기존", "신설"])
]

print("전체 대상 후보:", len(target))

print("\n지역별 후보 수")
print(target["지역"].value_counts().to_string())

print("\n학교구분별 후보 수")
print(target["학교구분"].value_counts().to_string())

print("\n후보 내 좌표 중복:", target.duplicated(["위도", "경도"]).sum())

print("\n대학교 중복 상세 검사 완료!")
