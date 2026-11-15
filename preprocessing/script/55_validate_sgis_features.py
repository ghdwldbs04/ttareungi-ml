
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

STATIONS = PROCESSED / "stations_clean.csv"
SGIS = PROCESSED / "sgis_features_new.csv"

FEATURES = [
    "grid_pop_total",
    "grid_households",
    "grid_housing",
    "grid_businesses",
    "grid_workers",
]

EXPECTED_MISSING = {
    "grid_pop_total": 26,
    "grid_households": 29,
    "grid_housing": 56,
    "grid_businesses": 11,
    "grid_workers": 11,
}

EXPECTED_CITIES = {
    "seoul": 2765,
    "daejeon": 1395,
    "sejong": 723,
    "changwon": 341,
}

print("=" * 70)
print("1. 저장된 파일 다시 읽기")

stations = pd.read_csv(
    STATIONS,
    dtype={"station": str},
    low_memory=False
)

sgis = pd.read_csv(
    SGIS,
    dtype={
        "station": str,
        "GRID_CD": str
    },
    low_memory=False
)

print("원본 대여소:", len(stations))
print("SGIS 피처:", len(sgis))
print("SGIS 열 수:", len(sgis.columns))

expected_cols = [
    "city",
    "station",
    "GRID_CD",
    *FEATURES,
    *[f"{f}_missing" for f in FEATURES],
    "grid_any_missing",
    "grid_all_missing",
]

if list(sgis.columns) != expected_cols:
    raise ValueError(
        "SGIS 열 구성이 예상과 다릅니다.\n"
        f"실제: {list(sgis.columns)}"
    )

if len(stations) != 5224 or len(sgis) != 5224:
    raise ValueError("대여소 수가 5224개가 아닙니다.")

print("열 구성: 정상")
print("행 수: 정상")

print("\n" + "=" * 70)
print("2. 대여소 키와 순서 검증")

station_keys = stations[
    ["city", "station"]
].reset_index(drop=True)

sgis_keys = sgis[
    ["city", "station"]
].reset_index(drop=True)

if not station_keys.equals(sgis_keys):
    raise ValueError(
        "원본 대여소와 SGIS 파일의 키 또는 순서가 다릅니다."
    )

if sgis.duplicated(
    ["city", "station"]
).any():
    raise ValueError("대여소 키 중복 발견")

if sgis["GRID_CD"].isna().any():
    raise ValueError("격자코드 결측 발견")

print("대여소 키 및 순서: 일치")
print("대여소 키 중복: 0")
print("격자코드 결측: 0")
print("사용 격자 수:", sgis["GRID_CD"].nunique())

if sgis["GRID_CD"].nunique() != 906:
    raise ValueError("사용 격자 수가 906개가 아닙니다.")

print("\n도시별 대여소 수:")
city_counts = sgis["city"].value_counts()

print(city_counts.to_string())

if city_counts.to_dict() != EXPECTED_CITIES:
    raise ValueError("도시별 대여소 수 불일치")

print("\n" + "=" * 70)
print("3. 통계값과 결측 표시 검증")

for feature in FEATURES:
    sgis[feature] = pd.to_numeric(
        sgis[feature],
        errors="raise"
    )

    missing_col = f"{feature}_missing"

    actual_missing = sgis[feature].isna().astype(int)

    if not sgis[missing_col].isin([0, 1]).all():
        raise ValueError(
            f"{missing_col}: 0/1 이외 값 발견"
        )

    if not (
        actual_missing.to_numpy()
        == sgis[missing_col].to_numpy()
    ).all():
        raise ValueError(
            f"{missing_col}: 실제 결측과 불일치"
        )

    missing_count = int(actual_missing.sum())

    if missing_count != EXPECTED_MISSING[feature]:
        raise ValueError(
            f"{feature}: 결측 수 불일치"
        )

    valid = sgis[feature].dropna()

    if not np.isfinite(valid.to_numpy()).all():
        raise ValueError(
            f"{feature}: 무한대 또는 비정상값"
        )

    if (valid < 0).any():
        raise ValueError(
            f"{feature}: 음수 통계값 발견"
        )

    print(
        f"{feature}: "
        f"결측 {missing_count}개, "
        f"결측 표시 일치, "
        f"값 범위 정상"
    )

print("\n" + "=" * 70)
print("4. 결측 요약 표시 검증")

expected_any = (
    sgis[FEATURES]
    .isna()
    .any(axis=1)
    .astype(int)
)

expected_all = (
    sgis[FEATURES]
    .isna()
    .all(axis=1)
    .astype(int)
)

if not (
    expected_any.to_numpy()
    == sgis["grid_any_missing"].to_numpy()
).all():
    raise ValueError("grid_any_missing 불일치")

if not (
    expected_all.to_numpy()
    == sgis["grid_all_missing"].to_numpy()
).all():
    raise ValueError("grid_all_missing 불일치")

print(
    "하나 이상 결측:",
    int(expected_any.sum())
)

print(
    "전부 결측:",
    int(expected_all.sum())
)

if int(expected_any.sum()) != 56:
    raise ValueError("하나 이상 결측 수 불일치")

if int(expected_all.sum()) != 11:
    raise ValueError("전부 결측 수 불일치")

print("\n" + "=" * 70)
print("5. 같은 격자 내부의 통계값 일관성")

inconsistent = []

for feature in FEATURES:
    # dropna=False: 결측도 하나의 고유 상태로 취급
    unique_counts = (
        sgis.groupby("GRID_CD")[feature]
        .nunique(dropna=False)
    )

    bad_grids = unique_counts[
        unique_counts > 1
    ]

    if not bad_grids.empty:
        inconsistent.append(
            (feature, len(bad_grids))
        )

    print(
        f"{feature}: "
        f"값이 서로 다른 격자 {len(bad_grids)}개"
    )

if inconsistent:
    raise ValueError(
        f"격자 내부 통계값 불일치: {inconsistent}"
    )

print("\n" + "=" * 70)
print("6. 최종 판정")

print("모든 검증 통과!")
print("SGIS 피처 파일:", SGIS.relative_to(BASE))
print("행 수:", len(sgis))
print("열 수:", len(sgis.columns))
print("사용 격자:", sgis["GRID_CD"].nunique())
print("기존 stations_clean.csv 수정 없음")

print("\nSGIS 피처 최종 검증 완료!")
