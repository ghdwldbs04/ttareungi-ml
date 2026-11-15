
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

# 전국의 운영 중이거나 신설된 대학 후보
candidates = df[
    df["학교구분"].isin(
        ["대학", "전문대학", "대학원대학"]
    )
    & df["학교상태"].isin(["기존", "신설"])
].copy()

print("=" * 60)
print("1. 전국 대학 후보")

print("전체 후보 수:", len(candidates))
print(
    "좌표 중복 행 수:",
    candidates.duplicated(["위도", "경도"]).sum()
)

print("\n" + "=" * 60)
print("2. 대상 지역의 좌표 중복 학교")

target = candidates[
    candidates["지역"].isin(
        ["서울", "대전", "세종", "경남"]
    )
].copy()

duplicates = target[
    target.duplicated(
        ["위도", "경도"],
        keep=False
    )
].sort_values(["위도", "경도", "학교명"])

print("중복 좌표에 속한 전체 행:", len(duplicates))

columns = [
    "학교명",
    "학교구분",
    "학제",
    "지역",
    "본분교",
    "학교상태",
    "위도",
    "경도",
    "도로명 주소"
]

if len(duplicates) > 0:
    print(duplicates[columns].to_string(index=False))
else:
    print("대상 지역에 좌표 중복 없음")

print("\n" + "=" * 60)
print("3. 전국 후보의 좌표 중복 유형")

national_duplicates = candidates[
    candidates.duplicated(
        ["위도", "경도"],
        keep=False
    )
].sort_values(["위도", "경도"])

print("중복 좌표에 속한 전체 행:", len(national_duplicates))
print(
    "중복 좌표 그룹 수:",
    national_duplicates.groupby(
        ["위도", "경도"]
    ).ngroups
)

print("\n전국 중복 학교 예시 (최대 30행)")
print(
    national_duplicates[columns]
    .head(30)
    .to_string(index=False)
)

print("\n대학교 좌표 중복 확인 완료!")
