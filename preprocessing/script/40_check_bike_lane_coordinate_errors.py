
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from pyproj import Transformer

BASE = Path(__file__).resolve().parent.parent
FOOTAGE = BASE / "footage"

files = [
    p for p in FOOTAGE.rglob("*.csv")
    if "전국자전거도로표준데이터"
    in unicodedata.normalize("NFC", p.name)
]

if len(files) != 1:
    raise ValueError(f"원본 파일 확인 필요: {files}")

df = pd.read_csv(
    files[0],
    encoding="cp949",
    low_memory=False
)

# 1. 서울·대전 자료 선택
sido = df["시도명"].fillna("").astype(str)

target = df[
    sido.str.contains("서울|대전", regex=True)
].copy()

target["city"] = np.where(
    target["시도명"].str.contains("서울"),
    "seoul",
    "daejeon"
)

# 2. 좌표와 길이 변환
coord_cols = [
    "기점위도", "기점경도",
    "종점위도", "종점경도"
]

for col in coord_cols:
    target[col] = pd.to_numeric(
        target[col],
        errors="coerce"
    )

target["length_m"] = pd.to_numeric(
    target["총길이(km)"],
    errors="coerce"
) * 1000

valid = (
    target["기점위도"].between(33, 39)
    & target["기점경도"].between(124, 132)
    & target["종점위도"].between(33, 39)
    & target["종점경도"].between(124, 132)
)

both = target[valid].copy()

# 3. 직선거리 계산
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5179",
    always_xy=True
)

sx, sy = transformer.transform(
    both["기점경도"].to_numpy(),
    both["기점위도"].to_numpy()
)

ex, ey = transformer.transform(
    both["종점경도"].to_numpy(),
    both["종점위도"].to_numpy()
)

both["straight_m"] = np.hypot(
    ex - sx,
    ey - sy
)

both["difference_m"] = (
    both["straight_m"] - both["length_m"]
)

# 4. 길이 차이 큰 기록
print("=" * 70)
print("1. 직선거리가 기록 길이보다 100m 이상 긴 기록")

suspicious = both[
    both["difference_m"] > 100
].copy()

print("전체:", len(suspicious))

print("\n도시별:")
print(
    suspicious["city"]
    .value_counts()
    .to_string()
)

print("\n차이가 가장 큰 기록 20개:")
cols = [
    "city",
    "노선명",
    "시군구명",
    "기점위도",
    "기점경도",
    "종점위도",
    "종점경도",
    "length_m",
    "straight_m",
    "difference_m"
]

print(
    suspicious[cols]
    .sort_values("difference_m", ascending=False)
    .head(20)
    .round(3)
    .to_string(index=False)
)

# 5. 직선거리 20km 초과
print("\n" + "=" * 70)
print("2. 기점·종점 직선거리 20km 초과")

very_long = both[
    both["straight_m"] > 20000
]

print("전체:", len(very_long))

print(
    very_long[cols]
    .sort_values("straight_m", ascending=False)
    .round(3)
    .to_string(index=False)
)

# 6. 같은 좌표로 입력된 기록
print("\n" + "=" * 70)
print("3. 기점·종점 좌표가 동일한 기록")

same_coords = both[
    (both["기점위도"] == both["종점위도"])
    & (both["기점경도"] == both["종점경도"])
]

print("전체:", len(same_coords))

print("\n도시별:")
print(
    same_coords["city"]
    .value_counts()
    .to_string()
)

print("\n예시 10개:")
print(
    same_coords[
        ["city", "노선명", "시군구명", "length_m"]
    ]
    .head(10)
    .to_string(index=False)
)

# 7. 도시별 좌표 범위
print("\n" + "=" * 70)
print("4. 도시별 좌표 범위")

for city, group in both.groupby("city"):
    print(f"\n[{city}]")

    for col in coord_cols:
        print(
            f"{col}: "
            f"{group[col].min():.6f} ~ "
            f"{group[col].max():.6f}"
        )

print("\n자전거도로 좌표 이상값 검사 완료!")
    