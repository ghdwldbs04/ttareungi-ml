
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"
PROCESSED = BASE / "processed"

# 1. 상가 원본 폴더 찾기
shop_dirs = [
    p for p in FOOTAGE.iterdir()
    if p.is_dir()
    and "상가" in unicodedata.normalize("NFC", p.name)
    and "20260930" in p.name
]

if len(shop_dirs) != 1:
    raise ValueError(f"상가 폴더를 확인해 주세요: {shop_dirs}")

SHOP_DIR = shop_dirs[0]

# 2. 대여소 데이터
stations = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str}
)

print("대여소 수:", len(stations))

# 3. 상가 원본 읽기
columns = [
    "상가업소번호",
    "시도명",
    "시군구명",
    "상권업종대분류명",
    "위도",
    "경도"
]

target_regions = {
    "서울특별시",
    "대전광역시",
    "세종특별자치시",
    "경상남도"
}

frames = []

for path in sorted(SHOP_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(
            path,
            usecols=columns,
            encoding="utf-8-sig",
            low_memory=False
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            usecols=columns,
            encoding="cp949",
            low_memory=False
        )

    df = df[df["시도명"].isin(target_regions)].copy()

    # 경남은 창원시만 사용
    df = df[
        (df["시도명"] != "경상남도")
        | df["시군구명"].fillna("").str.contains("창원")
    ].copy()

    if not df.empty:
        frames.append(df)
        print("읽기 완료:", path.name, len(df))

shops = pd.concat(frames, ignore_index=True)

# 4. 좌표 및 중복 정리
shops["위도"] = pd.to_numeric(
    shops["위도"], errors="coerce"
)
shops["경도"] = pd.to_numeric(
    shops["경도"], errors="coerce"
)

shops = shops.dropna(subset=["위도", "경도"])

shops = shops[
    shops["위도"].between(33, 39)
    & shops["경도"].between(124, 132)
].copy()

shops = shops.drop_duplicates(
    subset=["상가업소번호"]
).copy()

print("\n최종 상가 수:", len(shops))

# 5. 위경도를 미터 단위 좌표로 변환
# EPSG:4326 -> EPSG:5179
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

station_x, station_y = transformer.transform(
    stations["lon"].to_numpy(),
    stations["lat"].to_numpy()
)

shop_x, shop_y = transformer.transform(
    shops["경도"].to_numpy(),
    shops["위도"].to_numpy()
)

station_xy = np.column_stack([station_x, station_y])
shop_xy = np.column_stack([shop_x, shop_y])

# 6. 상가 주변 개수 계산
shop_tree = cKDTree(shop_xy)

result = stations[["city", "station"]].copy()

for radius in [300, 500, 1000]:
    result[f"shop_{radius}"] = shop_tree.query_ball_point(
        station_xy,
        r=radius,
        return_length=True
    )

    print(f"상가 반경 {radius}m 계산 완료")

# 7. 음식 업종 상가 수 계산
food_mask = (
    shops["상권업종대분류명"] == "음식"
).to_numpy()

food_xy = shop_xy[food_mask]
food_tree = cKDTree(food_xy)

result["food_500"] = food_tree.query_ball_point(
    station_xy,
    r=500,
    return_length=True
)

print("음식 업종 반경 500m 계산 완료")

# 8. 저장
output = PROCESSED / "shop_features_new.csv"

result.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n피처 요약")
print(
    result[
        ["shop_300", "shop_500", "shop_1000", "food_500"]
    ].describe()
)

print("\n저장 완료:", output)
print("저장된 대여소 수:", len(result))

print("\n처음 10개 대여소")
print(result.head(10).to_string(index=False))
