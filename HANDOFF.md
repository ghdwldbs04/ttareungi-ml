# 인수인계 메모 (전처리·변수 선택 담당용)

## 지금 상태
- 4개 도시 대여소 피처 완료: `data/processed/features_station_v2.csv` (5,224행)
  - 서울 2,765 / 대전 1,395 / 세종 723 / 창원 341
  - 목표변수: `log_y` = log(1 + daily_mean), `rank_pct` = 도시 안 백분위
  - 피처 22개: bus_300, bus_500, dist_subway_m, subway_1000, dist_univ_m, dist_park_m, park_500,
    park_area_500, shop_300, shop_500, food_500, pop_1km, young_share_1km, workers_1km, firms_1km,
    stations_500, dist_nn_station_m, dist_river_m, cycleway_m_500, dist_cycleway_m,
    apt_share_1km, oneperson_share_1km
  - (bike_pts_500, bike_km_500 는 표준데이터 공백으로 실제로는 안 씀)
- 세종은 이번에 새로 추가함 (2025년 1~12월, 건별 250만 건)
  - `scripts/prep_sejong.py`: 세종 건별 데이터 → data/processed/sejong/
  - `scripts/fetch_station_api.py`: 행안부 공영자전거 API로 대여소 좌표 받기 (개인 API 키 필요, zip에는 없음)
- 3개 도시 시절 백업: `features_backup_3city.py`, `features_station_3city.csv`, `features_station_v2_3city.csv`

## 전처리·변수 선택 때 알아둘 점
1. 결측: 세종 young_share_1km 22곳, apt_share_1km 33곳, oneperson_share_1km 24곳 (SGIS 격자값 없음)
2. 창원 자전거도로(cycleway_m_500, dist_cycleway_m)는 OSM 미등록이라 거의 0 → 실제 0 아님, 데이터 공백
3. 창원·세종은 도시철도가 없어 dist_subway_m 이 상한값 5000(=역 없음), subway_1000 = 0
4. 세종 dist_nn_station_m 중앙값 72m: 대여소(가상대여소 포함)가 매우 촘촘함
5. 인구·종사자·연령은 SGIS 1km 격자값(2024년) → 같은 격자 안 대여소는 값이 같음
6. 경사도, 거치대 수, 고령층 비율은 아직 없음 (추가 후보)
7. 개수·거리 변수는 치우침이 커서 Ridge 등 선형모델에는 log1p 변환 권장

## 실행 순서
eda_bike.py → features.py → features_extra.py → (모델링) experiments_v2/v3.py
원본 데이터(data/raw)는 용량 때문에 zip에서 뺐음. 피처를 처음부터 다시 계산하려면 따로 받아야 함.
