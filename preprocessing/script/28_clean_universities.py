
from pathlib import Path
import unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

files = [
    p for p in FOOTAGE.rglob("*.xlsx")
    if "대학교" in unicodedata.normalize("NFC", p.name)
    and "좌표" in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"대학교 원본 파일 확인 필요: {files}")

df = pd.read_excel(files[0])

print("원본 전체 행 수:", len(df))

# 1. 대학·전문대학·대학원대학만 선택
clean = df[
    df["학교구분"].isin(
        ["대학", "전문대학", "대학원대학"]
    )
].copy()

print("대학 유형 선택 후:", len(clean))

# 2. 기존·신설 학교만 선택
clean = clean[
    clean["학교상태"].isin(["기존", "신설"])
].copy()

print("폐교 제외 후:", len(clean))

# 3. 사이버대학 제외
cyber = clean["학제"].astype(str).str.contains(
    "사이버", na=False
)

print("사이버대학 제외 수:", cyber.sum())
clean = clean.loc[~cyber].copy()

# 4. 좌표 숫자 변환 및 유효성 검사
clean["위도"] = pd.to_numeric(
    clean["위도"], errors="coerce"
)
clean["경도"] = pd.to_numeric(
    clean["경도"], errors="coerce"
)

valid = (
    clean["위도"].between(33, 39)
    & clean["경도"].between(124, 132)
)

print("좌표 이상 제외 수:", (~valid).sum())
clean = clean.loc[valid].copy()

# 5. 같은 좌표는 캠퍼스 위치 기준으로 1개
before = len(clean)

clean = clean.drop_duplicates(
    subset=["위도", "경도"],
    keep="first"
).copy()

print("동일 좌표 추가 제외 수:", before - len(clean))

# 6. 필요한 열만 정리
columns = [
    "학교코드변환",
    "학교명",
    "학교구분",
    "학제",
    "지역",
    "본분교",
    "학교상태",
    "도로명 주소",
    "위도",
    "경도"
]

clean = clean[columns].reset_index(drop=True)

# 7. 저장
PROCESSED.mkdir(exist_ok=True)

output = PROCESSED / "universities_clean.csv"

clean.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 55)
print("최종 대학 캠퍼스 위치 수:", len(clean))

print("\n지역별 캠퍼스 위치 수")
print(clean["지역"].value_counts().to_string())

print("\n중복 좌표:", clean.duplicated(["위도", "경도"]).sum())
print("좌표 결측:", clean[["위도", "경도"]].isna().sum().sum())

print("\n저장 위치:", output)
print("대학교 1차 전처리 완료!")
