
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "processed"

# 기존 데이터와 새 데이터 읽기
old = pd.read_csv(
    DATA_DIR / "stations_clean.csv",
    encoding="utf-8-sig",
    dtype={"station": "string"}
)

new = pd.read_csv(
    DATA_DIR / "bus_features_new.csv",
    encoding="utf-8-sig",
    dtype={"station": "string"}
)

# 비교할 기존 버스 피처 자동 확인
candidates = ["bus_300", "bus_500", "bus_1000"]
available = [c for c in candidates if c in old.columns]

print("기존 데이터의 버스 피처:", available)

# 같은 대여소끼리 연결
merged = old[
    ["city", "station"] + available
].merge(
    new,
    on=["city", "station"],
    how="outer",
    suffixes=("_old", "_new"),
    indicator=True,
    validate="one_to_one"
)

print("\n1. 대여소 연결 결과")
print(merged["_merge"].value_counts().to_string())

print("\n2. 기존 값과 새 값 비교")

for col in available:
    old_col = f"{col}_old"
    new_col = f"{col}_new"

    a = pd.to_numeric(merged[old_col], errors="coerce")
    b = pd.to_numeric(merged[new_col], errors="coerce")

    valid = a.notna() & b.notna()
    diff = b[valid] - a[valid]

    print(f"\n[{col}]")
    print("비교 가능한 대여소:", valid.sum())
    print("값이 일치하는 대여소:", (diff == 0).sum())
    print("평균 차이(새 값 - 기존 값):", round(diff.mean(), 3))
    print("평균 절대 차이:", round(diff.abs().mean(), 3))
    print("최대 절대 차이:", diff.abs().max())
    print("상관계수:", round(a[valid].corr(b[valid]), 4))

print("\n버스 피처 비교 완료!")
