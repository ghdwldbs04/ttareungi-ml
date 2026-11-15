
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

OUTPUT = PROCESSED / "stations_features_v2.csv"

SOURCES = {
    "bus": "bus_features_new.csv",
    "subway": "subway_features_final.csv",
    "shop": "shop_features_new.csv",
    "park": "park_features_new.csv",
    "university": "university_features_new.csv",
    "sgis": "sgis_features_new.csv",
    "slope": "slope_features_new.csv",
}

KEYS = ["city", "station"]

EXCLUDE_COLUMNS = [
    "bike_pts_500",
    "bike_km_500",
    "cycleway_m_500",
    "dist_cycleway_m",
]

EXPECTED_ROWS = 5224

print("=" * 70)
print("1. 기본 데이터 읽기")

base = pd.read_csv(
    PROCESSED / "stations_clean.csv",
    dtype={"station": str},
    low_memory=False
)

if len(base) != EXPECTED_ROWS:
    raise ValueError("기본 데이터 행 수 불일치")

if base.duplicated(KEYS).any():
    raise ValueError("기본 데이터 대여소 키 중복")

if base[KEYS].isna().any().any():
    raise ValueError("기본 데이터 대여소 키 결측")

if OUTPUT.exists():
    raise FileExistsError(
        f"통합 파일이 이미 존재합니다: {OUTPUT}"
    )

print("기본 행 수:", len(base))
print("기본 열 수:", len(base.columns))

print("\n" + "=" * 70)
print("2. 자전거도로 검증 보류 피처 제외")

existing_exclusions = [
    col for col in EXCLUDE_COLUMNS
    if col in base.columns
]

result = base.drop(
    columns=existing_exclusions
).copy()

print("제외 열:", existing_exclusions)
print("제외 후 열 수:", len(result.columns))

print("\n" + "=" * 70)
print("3. 새 피처 순차 통합")

for name, filename in SOURCES.items():
    print("\n" + "-" * 60)
    print("통합 피처:", name)

    feature = pd.read_csv(
        PROCESSED / filename,
        dtype={"station": str},
        low_memory=False
    )

    if len(feature) != EXPECTED_ROWS:
        raise ValueError(
            f"{name}: 행 수 불일치"
        )

    if feature.duplicated(KEYS).any():
        raise ValueError(
            f"{name}: 키 중복"
        )

    if feature[KEYS].isna().any().any():
        raise ValueError(
            f"{name}: 키 결측"
        )

    feature_columns = [
        col for col in feature.columns
        if col not in KEYS
    ]

    if not feature_columns:
        raise ValueError(
            f"{name}: 피처 열 없음"
        )

    overlap = [
        col for col in feature_columns
        if col in result.columns
    ]

    additions = [
        col for col in feature_columns
        if col not in result.columns
    ]

    print("교체 열:", overlap)
    print("추가 열:", additions)

    # 겹치는 기존 열을 제거한 뒤 새 값으로 병합
    result = result.drop(
        columns=overlap
    )

    before_rows = len(result)

    result = result.merge(
        feature,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True
    )

    if len(result) != before_rows:
        raise ValueError(
            f"{name}: 병합 후 행 수 변경"
        )

    if not (
        result["_merge"] == "both"
    ).all():
        raise ValueError(
            f"{name}: 병합 누락"
        )

    result = result.drop(
        columns="_merge"
    )

    if result.duplicated(KEYS).any():
        raise ValueError(
            f"{name}: 병합 후 키 중복"
        )

    print("통합 후 행 수:", len(result))
    print("통합 후 열 수:", len(result.columns))

print("\n" + "=" * 70)
print("4. 최종 데이터 검증")

if len(result) != EXPECTED_ROWS:
    raise ValueError("최종 행 수 불일치")

if result.duplicated(KEYS).any():
    raise ValueError("최종 대여소 키 중복")

if not result[KEYS].equals(
    base[KEYS]
):
    raise ValueError(
        "기본 데이터와 대여소 키 또는 순서 불일치"
    )

remaining_exclusions = [
    col for col in EXCLUDE_COLUMNS
    if col in result.columns
]

if remaining_exclusions:
    raise ValueError(
        f"자전거도로 제외 열이 남음: {remaining_exclusions}"
    )

if len(result.columns) != len(
    set(result.columns)
):
    raise ValueError("중복 열 이름 발생")

# 타깃 값이 통합 중 바뀌지 않았는지 검사
for target in [
    "daily_mean",
    "log_y",
    "total",
    "active_days",
]:
    if target in base.columns:
        if target not in result.columns:
            raise ValueError(
                f"기존 타깃 관련 열 누락: {target}"
            )

        if not result[target].equals(
            base[target]
        ):
            raise ValueError(
                f"기존 타깃 관련 값 변경: {target}"
            )

print("행 수:", len(result))
print("열 수:", len(result.columns))
print("대여소 키 및 순서: 정상")
print("중복 열: 없음")
print("제외 대상 자전거도로 열: 없음")
print("기존 타깃 관련 값: 변경 없음")

print("\n" + "=" * 70)
print("5. 결측 현황")

missing = result.isna().sum()
missing = missing[missing > 0]

if len(missing):
    print(
        missing.sort_values(
            ascending=False
        ).to_string()
    )
else:
    print("결측 없음")

print("\n" + "=" * 70)
print("6. 도시별 대여소 수")

print(
    result["city"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "=" * 70)
print("7. 최종 파일 저장")

result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("저장 파일:", OUTPUT.relative_to(BASE))
print("저장 행 수:", len(result))
print("저장 열 수:", len(result.columns))

print("\n최종 피처 통합 완료!")
print("기존 파일 수정 없음")
