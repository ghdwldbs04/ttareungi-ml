# EDA 요약

분석 연도: 2025, 대여소 분석 기준: 운영 60일 이상

| 도시        | 기간                     | 사용 건수   | 하루 평균 대여   | 대여소 수(전체/60일+)   |   대여소 일평균 대여 중앙값 |   좌표 매칭률 | 제거: 날짜오류/연도밖/2분미만 제자리   |   제자리 반납 비율(%) |
|:------------|:-------------------------|:------------|:-----------------|:------------------------|----------------------------:|--------------:|:---------------------------------------|----------------------:|
| 서울 따릉이 | 2025-01~2025-12 (12개월) | 37,372,654  | 102,391          | 2,810 / 2,795           |                       28.04 |         0.989 | 0 / 0 / 0                              |                   7.2 |
| 대전 타슈   | 2025-12~2026-10 (11개월) | 7,771,232   | 23,198           | 1,406 / 1,395           |                       10.28 |         1     | 0 / 0 / 237,305                        |                   5.4 |
| 창원 누비자 | 2025-01~2025-12 (11개월) | 4,154,059   | 12,437           | 461 / 458               |                       16.82 |         0.74  | 0 / 0 / 265,004                        |                   6.7 |

## 추가 지표

city                          대전 타슈  서울 따릉이  창원 누비자
지표                                                 
반납-대여 불균형 |비율|>0.2 대여소 비중(%)    0.6     8.8    18.0
상위 10% 대여소의 대여 비중(%)           38.9    30.5    33.8
이용시간 중앙값(분, 2분 구간)             11.0    11.0    11.0

## 그림

- figures/01_daily_trend.png
- figures/02_weekday_hour_heatmap.png
- figures/03_hourly_profile_weekday_weekend.png
- figures/04_station_distribution_concentration.png
- figures/05_duration_distance.png
- figures/06_station_map_changwon.png
- figures/06_station_map_daejeon.png
- figures/06_station_map_seoul.png
- figures/07_grid250m_changwon.png
- figures/07_grid250m_daejeon.png
- figures/07_grid250m_seoul.png
- figures/08_monthly_intensity.png
- figures/09_imbalance.png