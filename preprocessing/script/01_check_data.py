
from pathlib import Path
import pandas as pd

# 프로젝트 폴더 위치
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "footage"

# CSV와 엑셀 파일 검색
files = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in [".csv", ".xlsx"]
    and not p.name.startswith("~$")
]

print(f"발견된 파일 수: {len(files)}")

for file in sorted(files):
    print("\n" + "=" * 60)
    print("파일:", file.relative_to(BASE_DIR))

    try:
        if file.suffix.lower() == ".csv":
            # 국내 공공데이터의 주요 인코딩 시도
            for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
                try:
                    df = pd.read_csv(
                        file,
                        encoding=encoding,
                        nrows=1000,
                        low_memory=False
                    )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("CSV 인코딩을 확인하지 못했습니다.")
                continue
        else:
            df = pd.read_excel(file, nrows=1000)

        print("열 개수:", len(df.columns))
        print("변수명:", list(df.columns))
        print("첫 1,000행 중 읽은 행 수:", len(df))
        print("샘플 결측값 개수:")
        print(df.isna().sum().to_string())

    except Exception as e:
        print("읽기 오류:", str(e))

print("\n원본 데이터 구조 확인 완료")
