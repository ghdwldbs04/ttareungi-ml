# -*- coding: utf-8 -*-
"""
신규 대여소 후보지(250m 격자) 피처 만들기 — 대전·창원
기존 대여소 피처(features.py + features_extra.py)와 똑같은 정의로 계산함

    python grid_features.py             # 대전·창원
    python grid_features.py changwon    # 한 도시만

후보지 조건: 구 경계 안 + 기존 대여소에서 200m 이상 1.5km 이내 (지금 대여소가 없지만 네트워크를 넓힐 수 있는 곳)
출력: data/processed/grid_candidates.csv
"""
import io
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from features import (to_utmk, PointIndex, nearest_dist, grid_code, cached, city_boxes, dec, FEAT, PROC, RADII,
                      src_bus, src_subway, src_univ, src_park, src_shop, src_sgis)
from features_extra import extract_layer, dbf_column, shp_lines, densify, nearest

CELL = 250
NEAR = 1500
MIN_GAP = 200
OUT = PROC / "grid_candidates.csv"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_station_features():
    return pd.read_csv(PROC / "features_station_v2.csv", dtype={"station": str})


def candidates(st, cities):
    from viz_map import load_gu, inside
    gus = load_gu()
    rows = []
    for c in cities:
        s = st[st.city == c]
        x0, x1, y0, y1 = s.x.min() - NEAR, s.x.max() + NEAR, s.y.min() - NEAR, s.y.max() + NEAR
        xs = np.arange(np.floor(x0 / CELL) * CELL, x1, CELL) + CELL / 2
        ys = np.arange(np.floor(y0 / CELL) * CELL, y1, CELL) + CELL / 2
        X, Y = [a.ravel() for a in np.meshgrid(xs, ys)]
        gu = np.full(len(X), None, dtype=object)
        for g in gus[c]:
            m = pd.isna(gu) & inside(g["path"], X, Y)
            gu[m] = g["name"]
        k = ~pd.isna(gu)
        X, Y, gu = X[k], Y[k], gu[k]
        d = nearest(X, Y, s.x.values, s.y.values)
        k = (d <= NEAR) & (d >= MIN_GAP)
        rows.append(pd.DataFrame({"city": c, "gu": gu[k], "x": X[k], "y": Y[k]}))
        log(f"  {c}: 후보 격자 {k.sum():,}개")
    return pd.concat(rows, ignore_index=True)


def features_at(P, st):
    boxes = city_boxes(st)
    src = {k: cached(k, lambda f=f: f(boxes)) for k, f in
           [("bus", src_bus), ("subway", src_subway), ("univ", src_univ), ("park", src_park),
            ("shop", src_shop), ("sgis", src_sgis)]}
    px, py = P.x.values, P.y.values
    F = pd.DataFrame(index=P.index)
    log("기본 피처")
    idx = PointIndex(src["bus"].x, src["bus"].y)
    for r in RADII:
        F[f"bus_{r}"] = idx.count_sum(px, py, r)[0]
    F["dist_subway_m"] = np.minimum(nearest_dist(px, py, src["subway"].x.values, src["subway"].y.values), 5000)
    F["subway_1000"] = PointIndex(src["subway"].x, src["subway"].y, cell=1000).count_sum(px, py, 1000)[0]
    F["dist_univ_m"] = nearest_dist(px, py, src["univ"].x.values, src["univ"].y.values)
    pk = src["park"]
    F["dist_park_m"] = nearest_dist(px, py, pk.x.values, pk.y.values)
    F["park_500"], F["park_area_500"] = PointIndex(pk.x, pk.y, pk["공원면적"]).count_sum(px, py, 500)
    sh = src["shop"]
    idx = PointIndex(sh.x, sh.y)
    food = sh[sh["상권업종대분류명"].astype(str).str.contains("음식")]
    for r in RADII:
        F[f"shop_{r}"] = idx.count_sum(px, py, r)[0]
    F["food_500"] = PointIndex(food.x, food.y).count_sum(px, py, 500)[0]
    g = src["sgis"].set_index("격자코드")
    codes = grid_code(px, py)
    F["pop_1km"] = g["pop"].reindex(codes).fillna(0).values
    F["young_share_1km"] = (g["young"].reindex(codes).values / F["pop_1km"].replace(0, np.nan)).clip(0, 1)
    F["workers_1km"] = g["workers"].reindex(codes).fillna(0).values
    F["firms_1km"] = g["firms"].reindex(codes).fillna(0).values
    F["stations_500"], F["dist_nn_station_m"] = 0.0, np.nan
    for c, gp in P.groupby("city"):
        s = st[st.city == c]
        ii = gp.index.values
        F.loc[ii, "stations_500"] = PointIndex(s.x, s.y).count_sum(gp.x.values, gp.y.values, 500)[0]
        F.loc[ii, "dist_nn_station_m"] = nearest(gp.x.values, gp.y.values, s.x.values, s.y.values)

    log("하천·자전거도로 (OSM)")
    boxes_ll = [(s.lon.min() - 0.05, s.lon.max() + 0.05, s.lat.min() - 0.05, s.lat.max() + 0.05)
                for c, s in st.groupby("city") if c in set(P.city)]
    osm = next(FEAT.glob("south-korea*free.shp.zip"))
    shp, shx, dbf = extract_layer(osm, "gis_osm_waterways_free_1")
    rx, ry, _ = densify(shp_lines(shp, shx, dbf_column(dbf, "fclass") == "river", boxes_ll))
    shp, shx, dbf = extract_layer(osm, "gis_osm_water_a_free_1")
    ax_, ay_, _ = densify(shp_lines(shp, shx, dbf_column(dbf, "fclass") == "river", boxes_ll))
    F["dist_river_m"] = np.minimum(nearest(px, py, np.r_[rx, ax_], np.r_[ry, ay_]), 5000)
    shp, shx, dbf = extract_layer(osm, "gis_osm_roads_free_1")
    cx, cy, cw = densify(shp_lines(shp, shx, dbf_column(dbf, "fclass") == "cycleway", boxes_ll))
    if len(cx):
        F["cycleway_m_500"] = PointIndex(cx, cy, cw).count_sum(px, py, 500)[1]
        F["dist_cycleway_m"] = np.minimum(nearest(px, py, cx, cy), 5000)
    else:
        F["cycleway_m_500"], F["dist_cycleway_m"] = 0.0, 5000.0

    log("아파트·1인가구 비율 (SGIS)")
    sg = next(FEAT.glob("*SGIS*.zip"))
    tiles = {c[:2] for c in codes}
    rows = []
    with zipfile.ZipFile(sg) as z:
        for i in z.infolist():
            m = re.search(r"2024년_(주택|가구)_(..)_1K\.csv$", dec(i.filename))
            if m and m.group(2) in tiles:
                raw = z.read(i.filename)
                for enc in ("utf-8-sig", "cp949"):
                    try:
                        rows.append(pd.read_csv(io.BytesIO(raw), encoding=enc, dtype={"격자코드": str}))
                        break
                    except UnicodeDecodeError:
                        continue
    s = pd.concat(rows)
    s["통계값"] = pd.to_numeric(s["통계값"], errors="coerce")
    piv = s.pivot_table(index="격자코드", columns="통계항목", values="통계값", aggfunc="sum")
    get = lambda k: piv[k].reindex(codes).values if k in piv.columns else np.full(len(codes), np.nan)
    F["apt_share_1km"] = (get("ho_gb_003") / get("to_ho_001")).clip(0, 1)
    F["oneperson_share_1km"] = (get("ga_sd_005") / get("to_ga_001")).clip(0, 1)
    return pd.concat([P, F], axis=1)


def build(cities=("daejeon", "changwon")):
    st = load_station_features()
    log("후보 격자 만들기")
    P = candidates(st, cities)
    out = features_at(P, st)
    n0 = len(out)
    out = out[(out.pop_1km + out.workers_1km) > 0].reset_index(drop=True)   # 사람도 일자리도 없는 격자(산·하천 등) 제외
    log(f"  인구·종사자가 있는 격자만: {n0:,} → {len(out):,}")
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    log(f"저장: {OUT} ({len(out):,}행)")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a in ("daejeon", "changwon")]
    build(tuple(args) if args else ("daejeon", "changwon"))
