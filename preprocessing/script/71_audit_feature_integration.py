
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

FILES = {
    "base": "stations_clean.csv",
    "bus": "bus_features_new.csv",
    "subway": "subway_features_final.csv",
    "shop": "shop_features_new.csv",
    "park": "park_features_new.csv",
    "university": "university_features_new.csv",
    "sgis": "sgis_features_new.csv",
    "slope": "slope_features_new.csv",
}

KEYS = ["city", "station"]

print("=" * 70)
print("1. 통합 대상 파일 존재 확인")

for name, filename in FILES.items():
    path = PROCESSED / filename

    print(
        f"{name:12s}",
        filename,
        "존재" if path.exists() else "없음"
    )

    if not path.exists():
        raise FileNotFoundError(path)

print("\n" + "=" * 70)
print("2. 파일별 기본 구조 확인")

data = {}

for name, filename in FILES.items():
    df = pd.read_csv(
        PROCESSED / filename,
        dtype={"station": str},
        low_memory=False
    )

    data[name] = df

    if not all(k in df.columns for k in KEYS):
        raise ValueError(
            f"{name}: city/station 열 누락"
        )

    duplicated = int(
        df.duplicated(KEYS).sum()
    )

    key_missing = int(
        df[KEYS].isna().any(axis=1).sum()
    )

    print("\n" + "-" * 60)
    print("파일:", name)
    print("행 수:", len(df))
    print("열 수:", len(df.columns))
    print("대여소 키 중복:", duplicated)
    print("대여소 키 결측:", key_missing)

    print("열 이름:")
    print(df.columns.tolist())

    print("결측이 있는 열:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing):
        print(missing.to_string())
    else:
        print("없음")

    if duplicated:
        raise ValueError(
            f"{name}: 중복 키 발생"
        )

    if key_missing:
        raise ValueError(
            f"{name}: 키 결측 발생"
        )

print("\n" + "=" * 70)
print("3. 기본 대여소와 행 수 및 키 비교")

base = data["base"]

print("기본 대여소:", len(base))

for name, df in data.items():
    if name == "base":
        continue

    same_count = len(df) == len(base)

    same_order = df[KEYS].equals(
        base[KEYS]
    )

    base_keys = pd.MultiIndex.from_frame(
        base[KEYS]
    )

    feature_keys = pd.MultiIndex.from_frame(
        df[KEYS]
    )

    missing_keys = int(
        (~base_keys.isin(feature_keys)).sum()
    )

    extra_keys = int(
        (~feature_keys.isin(base_keys)).sum()
    )

    print(
        f"{name:12s}",
        f"행 수 일치={same_count}",
        f"순서 일치={same_order}",
        f"누락 키={missing_keys}",
        f"추가 키={extra_keys}"
    )

    if not same_count:
        raise ValueError(
            f"{name}: 행 수 불일치"
        )

    if missing_keys or extra_keys:
        raise ValueError(
            f"{name}: 대여소 키 불일치"
        )

print("\n" + "=" * 70)
print("4. 기본 데이터와 새 피처의 중복 열")

base_columns = set(base.columns) - set(KEYS)

for name, df in data.items():
    if name == "base":
        continue

    feature_columns = set(
        df.columns
    ) - set(KEYS)

    overlap = sorted(
        base_columns & feature_columns
    )

    new_columns = sorted(
        feature_columns - base_columns
    )

    print("\n" + "-" * 60)
    print("피처:", name)

    print("기존 열과 겹치는 열:")
    print(overlap if overlap else "없음")

    print("새로 추가되는 열:")
    print(new_columns if new_columns else "없음")

print("\n" + "=" * 70)
print("5. 새 피처 파일끼리 중복되는 열")

feature_names = [
    name for name in FILES
    if name != "base"
]

found = False

for i, name1 in enumerate(feature_names):
    cols1 = set(
        data[name1].columns
    ) - set(KEYS)

    for name2 in feature_names[i + 1:]:
        cols2 = set(
            data[name2].columns
        ) - set(KEYS)

        overlap = sorted(cols1 & cols2)

        if overlap:
            found = True
            print(
                name1,
                "<->",
                name2,
                ":",
                overlap
            )

if not found:
    print("새 피처 파일끼리 중복 열 없음")

print("\n" + "=" * 70)
print("6. 최종 통합 전 주의할 기존 열")

candidates = [
    "bike_pts_500",
    "bike_km_500",
    "cycleway_m_500",
    "dist_cycleway_m",
    "daily_mean",
    "log_y",
]

for col in candidates:
    if col in base.columns:
        print(
            col,
            "| 결측:",
            int(base[col].isna().sum()),
            "| 자료형:",
            base[col].dtype
        )
    else:
        print(col, "| 없음")

print("\n통합 전 구조 점검 완료!")
print("파일 저장 및 수정 없음")
