
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "footage"

# 파일명 대신 CSV 내부의 열 이름으로 버스정류장 데이터 찾기
required = ["정류장번호", "위도", "경도"]
bus_files = []

for path in DATA_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() != ".csv":
        continue

    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            sample = pd.read_csv(
                path,
                encoding=encoding,
                nrows=3
            )

            if all(col in sample.columns for col in required):
                bus_files.append((path, encoding))

            break
        except UnicodeDecodeError:
            continue
        except Exception:
            break

print("발견한 버스정류장 CSV:", len(bus_files))

for path, encoding in bus_files:
    print("-", path.name)

if len(bus_files) != 1:
    raise ValueError(
        "버스정류장 CSV가 정확히 1개인지 확인해 주세요."
    )

file_path, encoding = bus_files[0]

df = pd.read_csv(
    file_path,
    encoding=encoding,
    low_memory=False,
    dtype={"정류장번호": "string"}
)

print("\n1. 전체 행 수:", len(df))
print("2. 변수명:", list(df.columns))

print("\n3. 필수 열 결측값")
print(df[required].isna().sum().to_string())

print("\n4. 정류장번호 중복 행 수")
print(df.duplicated(subset=["정류장번호"]).sum())

df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
df["경도"] = pd.to_numeric(df["경도"], errors="coerce")

invalid = (
    df["위도"].isna()
    | df["경도"].isna()
    | ~df["위도"].between(33, 39)
    | ~df["경도"].between(124, 132)
)

print("\n5. 좌표 결측 또는 범위 이상:", invalid.sum())

print("\n6. 좌표 범위")
print(df[["위도", "경도"]].describe().to_string())

if "도시명" in df.columns:
    print("\n7. 도시명 상위 20개")
    print(df["도시명"].value_counts().head(20).to_string())

print("\n버스정류장 원본 검사 완료!")
