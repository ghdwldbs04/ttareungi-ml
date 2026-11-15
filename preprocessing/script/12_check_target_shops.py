
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
import unicodedata

SHOP_DIR = next(
    p for p in (BASE / "footage").iterdir()
    if p.is_dir()
    and "20260930" in p.name
    and "상가" in unicodedata.normalize("NFC", p.name)
)

TARGETS = {"서울특별시", "대전광역시", "세종특별자치시", "경상남도"}

results = []

for path in sorted(SHOP_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=[
                "상가업소번호", "시도명", "시군구명",
                "상권업종대분류명", "위도", "경도"
            ],
            low_memory=False
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="cp949",
            usecols=[
                "상가업소번호", "시도명", "시군구명",
                "상권업종대분류명", "위도", "경도"
            ],
            low_memory=False
        )

    df = df[df["시도명"].isin(TARGETS)].copy()

    # 경남에서는 창원시만 사용
    df = df[
        (df["시도명"] != "경상남도")
        | df["시군구명"].fillna("").str.contains("창원")
    ].copy()

    if df.empty:
        continue

    results.append(df)
    print(f"{path.name}: {len(df):,}행")

shops = pd.concat(results, ignore_index=True)

print("\n전체 대상 상가 수:", len(shops))
print("상가업소번호 중복:", shops["상가업소번호"].duplicated().sum())
print("위도 결측:", shops["위도"].isna().sum())
print("경도 결측:", shops["경도"].isna().sum())

print("\n지역별 상가 수")
print(shops["시도명"].value_counts().to_string())

print("\n업종별 상가 수")
print(shops["상권업종대분류명"].value_counts().to_string())


# 위도·경도를 숫자로 변환
lat = pd.to_numeric(shops["위도"], errors="coerce")
lon = pd.to_numeric(shops["경도"], errors="coerce")

# 숫자가 아니거나 대한민국의 대략적인 좌표 범위를 벗어난 경우
invalid = (
    lat.isna()
    | lon.isna()
    | ~lat.between(33, 39)
    | ~lon.between(124, 132)
)

print("\n3. 상가 좌표 검사")
print("좌표 이상 상가 수:", invalid.sum())

print("\n위도 범위:", lat.min(), "~", lat.max())
print("경도 범위:", lon.min(), "~", lon.max())

if invalid.any():
    print("\n좌표 이상 상가 예시")
    print(
        shops.loc[
            invalid,
            ["시도명", "시군구명", "위도", "경도"]
        ].head(10).to_string(index=False)
    )

print("\n전체 상가 데이터 검사 완료!")
