# -*- coding: utf-8 -*-
"""
추가 피처: 하천 접근성, 자전거도로(OSM), 아파트·1인가구 비율(SGIS)

    python features_extra.py

입력: data/processed/features_station.csv
      data/raw/features/south-korea-*-free.shp.zip  (Geofabrik OpenStreetMap)
      data/raw/features/국가데이터처_SGIS 격자 통계 및 경계_*.zip
출력: data/processed/features_station_v2.csv  (기존 피처 + 추가 피처)
"""
import io
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from features import to_utmk, PointIndex, grid_code, dec

ROOT = Path(__file__).resolve().parent
FEAT = ROOT / "data" / "raw" / "features"
PROC = ROOT / "data" / "processed"
TMP = PROC / "osm_layers"
STEP = 25.0  # 선을 25m 간격 점으로 쪼개서 계산


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------
# 의존성 없는 shapefile 읽기 (선/면의 좌표 + dbf 한 컬럼)
# ------------------------------------------------------------
def extract_layer(zpath, layer):
    TMP.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for ext in ("shp", "shx", "dbf"):
            name = f"{layer}.{ext}"
            if not (TMP / name).exists():
                log(f"  압축 풀기: {name}")
                z.extract(name, TMP)
    return TMP / f"{layer}.shp", TMP / f"{layer}.shx", TMP / f"{layer}.dbf"


def dbf_column(dbf, col):
    raw = np.memmap(dbf, dtype=np.uint8, mode="r")
    nrec = int(raw[4:8].view("<u4")[0])
    hlen = int(raw[8:10].view("<u2")[0])
    rlen = int(raw[10:12].view("<u2")[0])
    pos, off = 32, 1
    while raw[pos] != 0x0D:
        name = bytes(raw[pos:pos + 11]).split(b"\x00")[0].decode()
        width = int(raw[pos + 16])
        if name == col:
            idx = hlen + np.arange(nrec)[:, None] * rlen + off + np.arange(width)[None]
            vals = np.ascontiguousarray(raw[idx]).view(f"S{width}").ravel()
            return np.char.strip(np.char.decode(vals, "utf-8", errors="ignore"))
        off += width
        pos += 32
    raise KeyError(col)


def shp_lines(shp, shx, keep, boxes_ll):
    """keep: 읽을 레코드 마스크. boxes_ll: [(lon0, lon1, lat0, lat1)] 이 범위와 겹치는 것만"""
    off = np.fromfile(shx, dtype=">i4", offset=100).reshape(-1, 2)[:, 0].astype(np.int64) * 2 + 8
    raw = np.memmap(shp, dtype=np.uint8, mode="r")
    bb = np.ascontiguousarray(raw[(off + 4)[:, None] + np.arange(32)[None]]).view("<f8").reshape(-1, 4)
    hit = np.zeros(len(off), bool)
    for lo0, lo1, la0, la1 in boxes_ll:
        hit |= (bb[:, 0] <= lo1) & (bb[:, 2] >= lo0) & (bb[:, 1] <= la1) & (bb[:, 3] >= la0)
    parts_out = []
    for o in off[hit & keep]:
        nparts = int(raw[o + 36:o + 40].view("<i4")[0])
        npts = int(raw[o + 40:o + 44].view("<i4")[0])
        parts = np.r_[raw[o + 44:o + 44 + 4 * nparts].view("<i4"), npts]
        p0 = o + 44 + 4 * nparts
        xy = np.asarray(raw[p0:p0 + 16 * npts]).view("<f8").reshape(-1, 2)
        for a, b in zip(parts[:-1], parts[1:]):
            parts_out.append(xy[a:b])
    return parts_out


def densify(parts):
    """선(위경도) -> UTM-K 25m 간격 점 + 각 점이 대표하는 길이(m)"""
    X, Y, W = [], [], []
    for p in parts:
        if len(p) < 2:
            continue
        x, y = to_utmk(p[:, 1], p[:, 0])
        dx, dy = np.diff(x), np.diff(y)
        L = np.hypot(dx, dy)
        n = np.maximum(np.ceil(L / STEP), 1).astype(int)
        seg = np.repeat(np.arange(len(L)), n)
        k = np.concatenate([(np.arange(m) + 0.5) / m for m in n])
        X.append(x[:-1][seg] + dx[seg] * k)
        Y.append(y[:-1][seg] + dy[seg] * k)
        W.append((L / n)[seg])
    if not X:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W)


def nearest(px, py, tx, ty):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(np.c_[tx, ty]).query(np.c_[px, py])[0]
    except ImportError:
        out = np.empty(len(px))
        for s in range(0, len(px), 100):
            d2 = (px[s:s + 100, None] - tx[None]) ** 2 + (py[s:s + 100, None] - ty[None]) ** 2
            out[s:s + 100] = np.sqrt(d2.min(axis=1))
        return out


# ------------------------------------------------------------
def main():
    df = pd.read_csv(PROC / "features_station.csv", dtype={"station": str})
    px, py = df.x.values, df.y.values
    boxes_ll = [(g.lon.min() - 0.05, g.lon.max() + 0.05, g.lat.min() - 0.05, g.lat.max() + 0.05) for _, g in df.groupby("city")]
    osm = next(FEAT.glob("south-korea*free.shp.zip"))
    log("OSM 파일:", osm.name)

    # 1) 하천: 수로(river) + 하천 면(river) 경계선
    shp, shx, dbf = extract_layer(osm, "gis_osm_waterways_free_1")
    fc = dbf_column(dbf, "fclass")
    rx, ry, _ = densify(shp_lines(shp, shx, fc == "river", boxes_ll))
    shp, shx, dbf = extract_layer(osm, "gis_osm_water_a_free_1")
    fc = dbf_column(dbf, "fclass")
    ax_, ay_, _ = densify(shp_lines(shp, shx, fc == "river", boxes_ll))
    tx, ty = np.r_[rx, ax_], np.r_[ry, ay_]
    log(f"  하천 점 {len(tx):,}개")
    df["dist_river_m"] = np.minimum(nearest(px, py, tx, ty), 5000)

    # 2) 자전거도로: roads 중 fclass == cycleway
    shp, shx, dbf = extract_layer(osm, "gis_osm_roads_free_1")
    fc = dbf_column(dbf, "fclass")
    cx, cy, cw = densify(shp_lines(shp, shx, fc == "cycleway", boxes_ll))
    log(f"  자전거도로 {cw.sum() / 1000:,.0f} km (세 도시 주변)")
    idx = PointIndex(cx, cy, cw)
    df["cycleway_m_500"] = idx.count_sum(px, py, 500)[1]
    df["dist_cycleway_m"] = np.minimum(nearest(px, py, cx, cy), 5000)

    # 3) SGIS 1km 격자: 아파트 비율, 1인가구 비율
    sg = next(FEAT.glob("*SGIS*.zip"))
    codes = grid_code(px, py)
    tiles = {c[:2] for c in codes}
    rows = []
    with zipfile.ZipFile(sg) as z:
        for i in z.infolist():
            n = dec(i.filename)
            m = re.search(r"2024년_(주택|가구)_(..)_1K\.csv$", n)
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
    df["apt_share_1km"] = (get("ho_gb_003") / get("to_ho_001")).clip(0, 1)
    df["oneperson_share_1km"] = (get("ga_sd_005") / get("to_ga_001")).clip(0, 1)

    out = PROC / "features_station_v2.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    new = ["dist_river_m", "cycleway_m_500", "dist_cycleway_m", "apt_share_1km", "oneperson_share_1km"]
    log(f"저장: {out}")
    summ = df.groupby("city")[new].median().round(3)
    summ.loc["서울 상관(log_y)"] = [df[df.city == "seoul"][[f, "log_y"]].corr(method="spearman").iloc[0, 1] for f in new]
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
