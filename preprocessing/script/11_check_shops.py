
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "footage"

# 상가 데이터의 필수 열
required = {
    "상가업소번호",
    "상권업종대분류명",
    "위도",
    "경도"
}

matches = []

# 파일명이 아니라 열 이름으로 상가 파일 찾기
for path in DATA_DIR.rglob("*.csv"):
    try:
        sample = pd.read_csv(
            path,
            nrows=5,
            encoding="utf-8-sig"
        )

        if required.issubset(sample.columns):
            matches.append(path)

    except UnicodeDecodeError:
        try:
            sample = pd.read_csv(
                path,
                nrows=5,
                encoding="cp949"
            )

            if required.issubset(sample.columns):
                matches.append(path)

        except Exception:
            pass

    except Exception:
        pass

print("=" * 50)
print("1. 발견한 상가 원본 파일 수:", len(matches))

for path in matches:
    print("-", path.name)

if not matches:
    raise ValueError("상가 원본 CSV 파일을 찾지 못했습니다.")

print("\n" + "=" * 50)
print("2. 파일별 앞의 1000행 검사")

for path in matches:
    try:
        df = pd.read_csv(
            path,
            nrows=1000,
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            nrows=1000,
            encoding="cp949"
        )

    print("\n파일:", path.name)
    print("샘플 행 수:", len(df))

    print("위도 결측:", df["위도"].isna().sum())
    print("경도 결측:", df["경도"].isna().sum())

    print(
        "상가업소번호 중복:",
        df["상가업소번호"].duplicated().sum()
    )

    print("\n업종 대분류 상위 10개")
    print(
        df["상권업종대분류명"]
        .value_counts()
        .head(10)
        .to_string()
    )

print("\n상가 원본 파일 1차 검사 완료!")
