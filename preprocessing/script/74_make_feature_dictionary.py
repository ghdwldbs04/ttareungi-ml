
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

INPUT_FILE = PROCESSED / "stations_features_v2.csv"
DICT_FILE = PROCESSED / "feature_dictionary_v2.csv"
MODEL_FILE = PROCESSED / "model_feature_columns_v2.csv"

print("=" * 70)
print("1. 최종 통합 데이터 읽기")

df = pd.read_csv(
    INPUT_FILE,
    dtype={"station": str},
    low_memory=False
)

if df.shape != (5224, 52):
    raise ValueError("통합 데이터 크기 확인 필요")

if DICT_FILE.exists() or MODEL_FILE.exists():
    raise FileExistsError(
        "설명서 파일이 이미 존재합니다. "
        "기존 파일을 확인해 주세요."
    )

print("데이터 크기:", df.shape)

print("\n" + "=" * 70)
print("2. 열별 설명 및 모델 입력 기준 설정")

DESCRIPTIONS = {
    "city": "도시 구분",
    "station": "대여소 식별자",
    "lat": "대여소 위도",
    "lon": "대여소 경도",
    "x": "대여소 투영 좌표 X",
    "y": "대여소 투영 좌표 Y",
    "daily_mean": "대여소 일평균 대여량(목표변수 후보)",
    "total": "집계 기간 총대여량",
    "active_days": "대여량 집계에 사용된 유효 일수",
    "log_y": "일평균 대여량의 로그 변환값(목표변수 후보)",
    "rank_pct": "대여 실적 기반 순위 백분위",
    "bus_300": "반경 300m 이내 버스정류장 수",
    "bus_500": "반경 500m 이내 버스정류장 수",
    "bus_1000": "반경 1000m 이내 버스정류장 수",
    "dist_subway_m": "원본 전체 도시철도 역사 좌표 중 최근접 거리(m)",
    "subway_500": "반경 500m 이내 통합 도시철도 역사 위치 수",
    "subway_1000": "반경 1000m 이내 통합 도시철도 역사 위치 수",
    "shop_300": "반경 300m 이내 상가 수",
    "shop_500": "반경 500m 이내 상가 수",
    "shop_1000": "반경 1000m 이내 상가 수",
    "food_500": "반경 500m 이내 음식업 상가 수",
    "dist_park_m": "최근접 공원 대표 좌표까지 거리(m)",
    "park_500": "반경 500m 이내 공원 대표 좌표 수",
    "park_area_500": "반경 500m 이내 대표 좌표를 가진 공원의 전체 면적 합",
    "dist_univ_m": "최근접 정제 대학교 캠퍼스 좌표까지 거리(m)",
    "univ_300": "반경 300m 이내 정제 대학교 캠퍼스 좌표 수",
    "univ_500": "반경 500m 이내 정제 대학교 캠퍼스 좌표 수",
    "univ_1000": "반경 1000m 이내 정제 대학교 캠퍼스 좌표 수",
    "GRID_CD": "SGIS 2024년 1km 격자 식별 코드",
    "grid_pop_total": "SGIS 2024년 해당 1km 격자의 총인구",
    "grid_households": "SGIS 2024년 해당 1km 격자의 총가구수",
    "grid_housing": "SGIS 2024년 해당 1km 격자의 총주택(거처)수",
    "grid_businesses": "SGIS 2024년 해당 1km 격자의 총사업체수",
    "grid_workers": "SGIS 2024년 해당 1km 격자의 총종사자수",
    "slope_value": "대여소 위치의 경사도 래스터 픽셀값(단위 미확인)",
    "slope_missing": "경사도 래스터 NoData 여부(1=결측)",
    "grid_pop_total_missing": "격자 총인구 결측 여부(1=결측)",
    "grid_households_missing": "격자 총가구수 결측 여부(1=결측)",
    "grid_housing_missing": "격자 총주택수 결측 여부(1=결측)",
    "grid_businesses_missing": "격자 총사업체수 결측 여부(1=결측)",
    "grid_workers_missing": "격자 총종사자수 결측 여부(1=결측)",
    "grid_any_missing": "SGIS 5개 통계 중 하나라도 결측인지 여부",
    "grid_all_missing": "SGIS 5개 통계가 모두 결측인지 여부",
    "pop_1km": "기존 반경 1km 인구 관련 피처(원래 산출 기준 확인 필요)",
    "young_share_1km": "기존 반경 1km 청년 비율 관련 피처(산출 기준 확인 필요)",
    "workers_1km": "기존 반경 1km 종사자 관련 피처(산출 기준 확인 필요)",
    "firms_1km": "기존 반경 1km 사업체 관련 피처(산출 기준 확인 필요)",
    "stations_500": "기존 반경 500m 대여소 관련 피처(산출 기준 확인 필요)",
    "dist_nn_station_m": "기존 최근접 다른 대여소 거리(m, 산출 기준 확인 필요)",
    "dist_river_m": "기존 하천 관련 거리 피처(m, 산출 기준 확인 필요)",
    "apt_share_1km": "기존 반경 1km 아파트 비율 관련 피처(산출 기준 확인 필요)",
    "oneperson_share_1km": "기존 반경 1km 1인 가구 비율 관련 피처(산출 기준 확인 필요)",
}

TARGET_OR_LEAKAGE = {
    "daily_mean",
    "log_y",
    "total",
    "active_days",
    "rank_pct",
}

IDENTIFIERS = {
    "station",
    "GRID_CD",
}

# 생성 근거가 명확하지 않은 기존 피처는
# 기본 모델 입력 후보에서 보류
REVIEW_REQUIRED = {
    "pop_1km",
    "young_share_1km",
    "workers_1km",
    "firms_1km",
    "stations_500",
    "dist_nn_station_m",
    "dist_river_m",
    "apt_share_1km",
    "oneperson_share_1km",
}

rows = []

for col in df.columns:
    if col in TARGET_OR_LEAKAGE:
        role = "목표변수/누수 주의"
        use_model = "N"

    elif col in IDENTIFIERS:
        role = "식별자"
        use_model = "N"

    elif col in REVIEW_REQUIRED:
        role = "산출 근거 확인 필요"
        use_model = "보류"

    elif col == "city":
        role = "범주형 입력 후보"
        use_model = "Y"

    elif col.endswith("_missing"):
        role = "결측 표시 입력 후보"
        use_model = "Y"

    else:
        role = "수치형 입력 후보"
        use_model = "Y"

    rows.append({
        "column": col,
        "description": DESCRIPTIONS.get(
            col,
            "설명 추가 확인 필요"
        ),
        "dtype": str(df[col].dtype),
        "missing_count": int(df[col].isna().sum()),
        "role": role,
        "model_input_recommended": use_model,
    })

dictionary = pd.DataFrame(rows)

if dictionary["column"].duplicated().any():
    raise ValueError("설명서 열 중복")

if len(dictionary) != len(df.columns):
    raise ValueError("설명서 열 수 불일치")

unexplained = dictionary[
    dictionary["description"] == "설명 추가 확인 필요"
]

if len(unexplained):
    print("설명이 추가로 필요한 열:")
    print(unexplained["column"].tolist())
    raise ValueError("모든 열의 설명을 확인해 주세요")

print("설명서 열 수:", len(dictionary))

print("\n" + "=" * 70)
print("3. 모델 입력 후보 목록 생성")

model_columns = dictionary[
    dictionary["model_input_recommended"] == "Y"
][
    [
        "column",
        "description",
        "dtype",
        "missing_count"
    ]
].copy()

print("모델 입력 후보:", len(model_columns))
print("제외/보류:", len(dictionary) - len(model_columns))

print("\n역할별 열 수:")
print(
    dictionary["role"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 70)
print("4. 결과 파일 저장")

dictionary.to_csv(
    DICT_FILE,
    index=False,
    encoding="utf-8-sig"
)

model_columns.to_csv(
    MODEL_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("저장:", DICT_FILE.relative_to(BASE))
print("저장:", MODEL_FILE.relative_to(BASE))

print("\n" + "=" * 70)
print("5. 팀원 전달 시 주의사항")

print("목표변수 후보: daily_mean 또는 log_y")
print("두 목표변수를 동시에 입력 피처로 사용하지 말 것")
print("total, active_days, rank_pct는 기본 입력에서 제외")
print("city는 범주형 인코딩 필요")
print("결측은 모델에 맞춰 학습 데이터 기준으로 처리")
print("기존 출처 불명 피처 9개는 사용 보류")
print("2026년 상권/공원 자료와 2025년 대여량의 시점 차이 주의")
print("경사도 slope_value의 물리적 단위 미확인")

print("\n피처 설명서 생성 완료!")
print("기존 통합 파일 수정 없음")
