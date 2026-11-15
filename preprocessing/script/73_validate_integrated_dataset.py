
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

KEYS = ["city", "station"]

SOURCES = {
    "bus": "bus_features_new.csv",
    "subway": "subway_features_final.csv",
    "shop": "shop_features_new.csv",
    "park": "park_features_new.csv",
    "university": "university_features_new.csv",
    "sgis": "sgis_features_new.csv",
    "slope": "slope_features_new.csv",
}

EXCLUDED = [
    "bike_pts_500",
    "bike_km_500",
    "cycleway_m_500",
    "dist_cycleway_m",
]


def read_csv(filename):
    return pd.read_csv(
        PROCESSED / filename,
        dtype={"station": str},
        low_memory=False
    )


def compare_columns(left, right, columns, label):
    errors = []

    for col in columns:
        if col not in left.columns:
            errors.append(f"{col}: 통합 파일에 열 없음")
            continue

        if col not in right.columns:
            errors.append(f"{col}: 비교 파일에 열 없음")
            continue

        a = left[col]
        b = right[col]

        if not a.isna().equals(b.isna()):
            errors.append(f"{col}: 결측 위치 불일치")
            continue

        if (
            pd.api.types.is_numeric_dtype(a)
            and pd.api.types.is_numeric_dtype(b)
        ):
            same = np.isclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-8,
                equal_nan=True
            )

            if not same.all():
                errors.append(
                    f"{col}: 값 불일치 "
                    f"{int((~same).sum())}개"
                )
        else:
            if not a.equals(b):
                errors.append(f"{col}: 값 불일치")

    if errors:
        print(f"[실패] {label}")

        for error in errors:
            print(" -", error)

        raise ValueError(
            f"{label}: 값 일치 검증 실패"
        )

    print(
        f"[통과] {label}: {len(columns)}개 열 일치"
    )


print("=" * 70)
print("1. 저장된 통합 파일 다시 읽기")

final = read_csv("stations_features_v2.csv")
base = read_csv("stations_clean.csv")

print("통합 데이터:", final.shape)
print("기본 데이터:", base.shape)

if final.shape != (5224, 52):
    raise ValueError("최종 파일 크기 불일치")

if final.duplicated(KEYS).any():
    raise ValueError("통합 데이터 키 중복")

if final[KEYS].isna().any().any():
    raise ValueError("통합 데이터 키 결측")

if not final[KEYS].equals(base[KEYS]):
    raise ValueError("대여소 키 또는 순서 불일치")

if len(final.columns) != len(set(final.columns)):
    raise ValueError("중복 열 이름")

print("[통과] 행 수, 열 수, 대여소 키, 순서")

print("\n" + "=" * 70)
print("2. 새 피처 파일과 전체 값 비교")

replaced_columns = set()

for name, filename in SOURCES.items():
    feature = read_csv(filename)

    if len(feature) != len(final):
        raise ValueError(f"{name}: 행 수 불일치")

    if not feature[KEYS].equals(final[KEYS]):
        raise ValueError(
            f"{name}: 키 또는 순서 불일치"
        )

    columns = [
        col for col in feature.columns
        if col not in KEYS
    ]

    compare_columns(
        final,
        feature,
        columns,
        name
    )

    replaced_columns.update(columns)

print("\n" + "=" * 70)
print("3. 유지된 기본 데이터 열 비교")

preserved_columns = [
    col for col in base.columns
    if col not in KEYS
    and col not in replaced_columns
    and col not in EXCLUDED
]

compare_columns(
    final,
    base,
    preserved_columns,
    "기존 유지 열"
)

print("\n" + "=" * 70)
print("4. 제외 대상 열 확인")

remaining = [
    col for col in EXCLUDED
    if col in final.columns
]

print("남아 있는 제외 대상 열:", remaining)

if remaining:
    raise ValueError("자전거도로 제외 실패")

print("[통과] 검증 보류 자전거도로 열 제외")

print("\n" + "=" * 70)
print("5. 결측 표시 열 검증")

flag_pairs = [
    ("grid_pop_total", "grid_pop_total_missing"),
    ("grid_households", "grid_households_missing"),
    ("grid_housing", "grid_housing_missing"),
    ("grid_businesses", "grid_businesses_missing"),
    ("grid_workers", "grid_workers_missing"),
    ("slope_value", "slope_missing"),
]

for value_col, flag_col in flag_pairs:
    actual_missing = (
        final[value_col].isna().astype(int)
    )

    recorded_missing = final[
        flag_col
    ].astype(int)

    if not actual_missing.equals(recorded_missing):
        raise ValueError(
            f"{flag_col}: 결측 표시 불일치"
        )

    print(
        f"[통과] {flag_col}: "
        f"결측 {int(actual_missing.sum())}개"
    )

grid_value_cols = [
    "grid_pop_total",
    "grid_households",
    "grid_housing",
    "grid_businesses",
    "grid_workers",
]

any_missing = (
    final[grid_value_cols]
    .isna()
    .any(axis=1)
    .astype(int)
)

all_missing = (
    final[grid_value_cols]
    .isna()
    .all(axis=1)
    .astype(int)
)

if not any_missing.equals(
    final["grid_any_missing"].astype(int)
):
    raise ValueError("grid_any_missing 불일치")

if not all_missing.equals(
    final["grid_all_missing"].astype(int)
):
    raise ValueError("grid_all_missing 불일치")

print("[통과] SGIS 전체 결측 표시")

print("\n" + "=" * 70)
print("6. 주요 수치 범위 검사")

nonnegative_columns = [
    "bus_300",
    "bus_500",
    "bus_1000",
    "dist_subway_m",
    "subway_500",
    "subway_1000",
    "shop_300",
    "shop_500",
    "shop_1000",
    "food_500",
    "dist_park_m",
    "park_500",
    "park_area_500",
    "dist_univ_m",
    "univ_300",
    "univ_500",
    "univ_1000",
    "grid_pop_total",
    "grid_households",
    "grid_housing",
    "grid_businesses",
    "grid_workers",
    "slope_value",
]

for col in nonnegative_columns:
    values = pd.to_numeric(
        final[col],
        errors="coerce"
    )

    if (
        values.notna()
        & ~np.isfinite(values)
    ).any():
        raise ValueError(
            f"{col}: 무한대 값"
        )

    if (values.dropna() < 0).any():
        raise ValueError(
            f"{col}: 음수 값"
        )

print("[통과] 주요 수치 열 음수 및 무한대 없음")

print("\n" + "=" * 70)
print("7. 최종 요약")

print("최종 행 수:", len(final))
print("최종 열 수:", len(final.columns))
print("대여소 키 중복:", int(final.duplicated(KEYS).sum()))

missing = final.isna().sum()
missing = missing[missing > 0]

print("결측 열 수:", len(missing))
print("결측 현황:")

if len(missing):
    print(missing.to_string())
else:
    print("없음")

print("\n최종 통합 데이터 검증 완료!")
print("파일 저장 및 수정 없음")
