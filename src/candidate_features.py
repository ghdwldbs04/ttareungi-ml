"""
신규 대여소 후보지 피처 생성 (Phase 4 준비)

조원의 v2 피처 코드(feature.zip: 04·07·13·19·20·28·29·54·60·70번)를 그대로 옮겨,
① 기존 대여소 5,224곳의 피처를 다시 계산해 stations_features_v2.csv와 일치하는지 검증하고
② 같은 코드로 후보지(대여소 없는 250m 격자) 피처를 만든다.

후보지 조건 (예전 grid_features.py와 동일)
  - 대전·세종·창원 행정구역 경계 안
  - 같은 도시 기존 대여소에서 200m 이상 1.5km 이내 (지금 대여소는 없지만 네트워크를 넓힐 수 있는 곳)

필요 파일
  data/raw/features/   버스·지하철·상가·공원·대학·SGIS 원본 (이미 있음)
  data/raw/boundary/skorea-municipalities-2018-geo.json (이미 있음)
  data/raw/slope/      경사도 래스터 Slope_All.tif (환경 빅데이터 플랫폼 '지형 경사도'에서 받아 압축 해제)

실행
  python src/candidate_features.py --validate   # 검증만 (경사도 래스터 없으면 경사 제외하고 검증)
  python src/candidate_features.py              # 검증 + 후보지 피처 생성 (경사도 래스터 필요)

결과
  results/candidate_features/validation_v2.csv   피처별 일치 여부
  data/candidates_features_v2.csv                후보지 피처

필요 패키지 : pandas numpy scipy pyproj geopandas rasterio openpyxl
"""
import argparse
import io
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "features"
SLOPE_DIR = DATA / "raw" / "slope"
BOUNDARY = DATA / "raw" / "boundary" / "skorea-municipalities-2018-geo.json"
SGIS_DIR = RAW / "_sgis_extracted"
OUT_VAL = ROOT / "results" / "candidate_features"
OUT_CAND = DATA / "candidates_features_v2.csv"
OUT_VAL.mkdir(parents=True, exist_ok=True)

CELL, NEAR, MIN_GAP = 250, 1500, 200
CITY_CODE = {"daejeon": "25", "sejong": "29", "changwon": "3811"}   # 2018 시군구 코드 앞자리
COUNT_COLS = ["bus_300", "bus_500", "bus_1000", "subway_500", "subway_1000", "shop_300", "shop_500",
              "shop_1000", "food_500", "park_500", "univ_300", "univ_500", "univ_1000"]
FLOAT_COLS = ["dist_subway_m", "dist_park_m", "park_area_500", "dist_univ_m", "grid_pop_total",
              "grid_households", "grid_housing", "grid_businesses", "grid_workers", "slope_value"]
SGIS_CODES = {
    "grid_pop_total": ("인구", "to_in_001"),
    "grid_households": ("가구", "to_ga_001"),
    "grid_housing": ("주택", "to_ho_001"),
    "grid_businesses": ("사업체", "to_fa_010"),
    "grid_workers": ("종사자", "to_em_020"),
}


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


def zname(info):
    """zip 안 한글 파일명 복원 (UTF-8 플래그 없으면 cp949)"""
    if info.flag_bits & 0x800:
        return nfc(info.filename)
    try:
        return nfc(info.filename.encode("cp437").decode("cp949"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return nfc(info.filename)


def find_one(folder, pattern, keyword):
    hits = [p for p in folder.glob(pattern) if keyword in nfc(p.name) and not p.name.startswith("~$")]
    if len(hits) != 1:
        raise FileNotFoundError(f"{folder}에서 '{keyword}' 파일을 정확히 1개 찾지 못함: {hits}")
    return hits[0]


def to_5179(lon, lat):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    x, y = t.transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack([x, y])


def read_csv_enc(open_fn, **kw):
    """조원 코드와 같은 순서로 인코딩 시도: utf-8-sig → cp949"""
    try:
        with open_fn() as f:
            return pd.read_csv(f, encoding="utf-8-sig", **kw)
    except UnicodeDecodeError:
        with open_fn() as f:
            return pd.read_csv(f, encoding="cp949", **kw)


# ================================================================ 원본 소스 (조원 코드 그대로)
def src_bus():                                                   # 04번
    path = find_one(RAW, "*.csv", "버스정류장")
    bus = read_csv_enc(lambda: open(path, "rb"), low_memory=False, dtype={"정류장번호": "string"})
    for c in ["위도", "경도"]:
        bus[c] = pd.to_numeric(bus[c], errors="coerce")
    bus = bus[bus["위도"].between(33, 39) & bus["경도"].between(124, 132)]
    return to_5179(bus["경도"], bus["위도"])


def src_subway():                                                # 07번(개수) + 70번(거리)
    raw = pd.read_excel(find_one(RAW, "*.xlsx", "도시철도역사정보"))
    raw_xy = to_5179(raw["역경도"], raw["역위도"])               # 70번: 원본 전체 역사로 거리

    s = raw.copy()
    s["역위도"] = pd.to_numeric(s["역위도"], errors="coerce")
    s["역경도"] = pd.to_numeric(s["역경도"], errors="coerce")
    s = s[s["역위도"].between(33, 39) & s["역경도"].between(124, 132)].copy()
    xy = to_5179(s["역경도"], s["역위도"])
    s["x_m"], s["y_m"] = xy[:, 0], xy[:, 1]
    s = s.sort_values(["역사명", "x_m", "y_m"]).reset_index(drop=True)
    keep = []
    for _, g in s.groupby("역사명"):                             # 같은 이름 + 100m 이내 → 1개
        kept = []
        for idx, row in g.iterrows():
            p = np.array([row["x_m"], row["y_m"]])
            if not kept or min(np.linalg.norm(p - q) for q in kept) > 100:
                keep.append(idx)
                kept.append(p)
    u = s.loc[keep]
    return raw_xy, np.column_stack([u["x_m"].values, u["y_m"].values])


def src_shop():                                                  # 13번
    zpath = find_one(RAW, "*.zip", "상가")
    cols = ["상가업소번호", "시도명", "시군구명", "상권업종대분류명", "위도", "경도"]
    regions = {"서울특별시", "대전광역시", "세종특별자치시", "경상남도"}
    frames = []
    with zipfile.ZipFile(zpath) as z:
        for info in sorted(z.infolist(), key=zname):
            name = zname(info)
            if not name.lower().endswith(".csv") or "__MACOSX" in name:
                continue
            if not any(k in name for k in ("서울", "대전", "세종", "경남")):
                continue    # 다른 시도 파일은 시도명 필터에서 어차피 전부 제외됨 → 읽기 생략 (검증으로 동일성 확인)
            df = read_csv_enc(lambda: z.open(info), usecols=cols, low_memory=False)
            df = df[df["시도명"].isin(regions)]
            df = df[(df["시도명"] != "경상남도") | df["시군구명"].fillna("").str.contains("창원")]
            if not df.empty:
                frames.append(df)
    shops = pd.concat(frames, ignore_index=True)
    shops["위도"] = pd.to_numeric(shops["위도"], errors="coerce")
    shops["경도"] = pd.to_numeric(shops["경도"], errors="coerce")
    shops = shops.dropna(subset=["위도", "경도"])
    shops = shops[shops["위도"].between(33, 39) & shops["경도"].between(124, 132)]
    shops = shops.drop_duplicates(subset=["상가업소번호"])
    xy = to_5179(shops["경도"], shops["위도"])
    return xy, xy[(shops["상권업종대분류명"] == "음식").to_numpy()]


def src_park():                                                  # 19번
    path = find_one(RAW, "*.csv", "전국도시공원정보")
    p = read_csv_enc(lambda: open(path, "rb"), low_memory=False)
    for c in ["위도", "경도", "공원면적"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[p["위도"].between(33, 39) & p["경도"].between(124, 132)]
    p = p.drop_duplicates(subset=["공원명", "공원구분", "위도", "경도", "공원면적"], keep="first")
    return to_5179(p["경도"], p["위도"]), p["공원면적"].to_numpy()


def src_univ():                                                  # 28번
    df = pd.read_excel(find_one(RAW, "*.xlsx", "대학교"))
    c = df[df["학교구분"].isin(["대학", "전문대학", "대학원대학"])]
    c = c[c["학교상태"].isin(["기존", "신설"])]
    c = c[~c["학제"].astype(str).str.contains("사이버", na=False)].copy()
    c["위도"] = pd.to_numeric(c["위도"], errors="coerce")
    c["경도"] = pd.to_numeric(c["경도"], errors="coerce")
    c = c[c["위도"].between(33, 39) & c["경도"].between(124, 132)]
    c = c.drop_duplicates(subset=["위도", "경도"], keep="first")
    return to_5179(c["경도"], c["위도"])


def ensure_sgis():
    if SGIS_DIR.exists() and any(SGIS_DIR.rglob("*.shp")):
        return SGIS_DIR
    zpath = find_one(RAW, "*.zip", "SGIS")
    log(f"SGIS zip 압축 해제 → {SGIS_DIR}")
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = zname(info)
            if info.is_dir() or "__MACOSX" in name:
                continue
            out = SGIS_DIR / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
    return SGIS_DIR


def sgis_features(lon, lat):                                     # 54번
    import geopandas as gpd
    base = ensure_sgis()
    shp = sorted(base.rglob("*.shp"), key=lambda p: nfc(str(p)))
    parts = []
    for p in shp:
        g = gpd.read_file(p)
        if g.crs is None or g.crs.to_epsg() != 5179:
            raise ValueError(f"SGIS 좌표계 확인 필요: {p.name}")
        parts.append(g[["GRID_CD", "geometry"]])
    grids = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:5179")
    grids["GRID_CD"] = grids["GRID_CD"].astype(str).str.strip()

    pts = gpd.GeoDataFrame({"_row": np.arange(len(lon))}, geometry=gpd.points_from_xy(lon, lat),
                           crs="EPSG:4326").to_crs("EPSG:5179")
    m = gpd.sjoin(pts, grids, how="left", predicate="intersects")
    m = m.drop_duplicates("_row").sort_values("_row")
    out = pd.DataFrame({"GRID_CD": m["GRID_CD"].values})

    stat_dirs = [d for d in base.rglob("*") if d.is_dir() and nfc(d.name) == "1. 통계"]
    all_csv = [p for d in stat_dirs for p in d.rglob("*.csv")]
    for feat, (topic, code) in SGIS_CODES.items():
        pieces = []
        for p in sorted([p for p in all_csv if f"_{topic}_" in nfc(p.name)], key=lambda p: nfc(str(p))):
            df = pd.read_csv(p, encoding="cp949", dtype=str, usecols=["기준연도", "격자코드", "통계항목", "통계값"])
            sel = df[(df["기준연도"].str.strip() == "2024") & (df["통계항목"].str.strip() == code)].copy()
            if sel.empty:
                continue
            sel["GRID_CD"] = sel["격자코드"].str.strip()
            sel[feat] = pd.to_numeric(sel["통계값"].str.strip(), errors="coerce")
            pieces.append(sel[["GRID_CD", feat]])
        comb = pd.concat(pieces, ignore_index=True)
        out = out.merge(comb, on="GRID_CD", how="left", validate="many_to_one")
    cols = list(SGIS_CODES)
    for c in cols:
        out[f"{c}_missing"] = out[c].isna().astype(int)
    out["grid_any_missing"] = out[cols].isna().any(axis=1).astype(int)
    out["grid_all_missing"] = out[cols].isna().all(axis=1).astype(int)
    return out


def slope_raster():
    if not SLOPE_DIR.exists():
        return None
    hits = [p for p in SLOPE_DIR.rglob("*.tif") if nfc(p.name).lower() == "slope_all.tif"]
    return hits[0] if len(hits) == 1 else None


def slope_features(lon, lat, raster):                            # 60번
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(raster) as src:
        xs, ys = transform("EPSG:4326", src.crs, list(lon), list(lat))
        vals = [np.nan if np.ma.is_masked(v[0]) else float(v[0])
                for v in src.sample(list(zip(xs, ys)), indexes=1, masked=True)]
    v = np.array(vals)
    return pd.DataFrame({"slope_value": v, "slope_missing": np.isnan(v).astype(int)})


# ================================================================ 피처 계산 (조원 코드와 같은 정의)
def compute_features(lon, lat, raster):
    pts = to_5179(lon, lat)
    F = pd.DataFrame(index=range(len(pts)))

    log("  버스")
    t = cKDTree(src_bus())
    for r in (300, 500, 1000):
        F[f"bus_{r}"] = t.query_ball_point(pts, r=r, return_length=True)

    log("  지하철")
    raw_xy, uniq_xy = src_subway()
    F["dist_subway_m"] = cKDTree(raw_xy).query(pts, k=1)[0]
    t = cKDTree(uniq_xy)
    for r in (500, 1000):
        F[f"subway_{r}"] = t.query_ball_point(pts, r=r, return_length=True)

    log("  상가 (원본이 커서 몇 분 걸림)")
    shop_xy, food_xy = src_shop()
    t = cKDTree(shop_xy)
    for r in (300, 500, 1000):
        F[f"shop_{r}"] = t.query_ball_point(pts, r=r, return_length=True)
    F["food_500"] = cKDTree(food_xy).query_ball_point(pts, r=500, return_length=True)

    log("  공원")                                                # 20번
    park_xy, area = src_park()
    t = cKDTree(park_xy)
    F["dist_park_m"] = t.query(pts, k=1)[0]
    near = t.query_ball_point(pts, r=500)
    F["park_500"] = [len(i) for i in near]
    F["park_area_500"] = [area[i].sum() for i in near]

    log("  대학")                                                # 29번
    t = cKDTree(src_univ())
    F["dist_univ_m"] = t.query(pts, k=1)[0]
    for r in (300, 500, 1000):
        F[f"univ_{r}"] = t.query_ball_point(pts, r=r, return_length=True)

    log("  SGIS 격자")
    F = pd.concat([F, sgis_features(lon, lat)], axis=1)

    if raster is not None:
        log("  경사도")
        F = pd.concat([F, slope_features(lon, lat, raster)], axis=1)
    else:
        F["slope_value"], F["slope_missing"] = np.nan, 1
    return F


# ================================================================ 후보지 격자
def make_candidates(st):
    import geopandas as gpd
    gu = gpd.read_file(BOUNDARY).to_crs("EPSG:5179")
    gu["code"] = gu["code"].astype(str)
    rows = []
    for city, prefix in CITY_CODE.items():
        s = st[st.city == city]
        x0, x1 = s.x.min() - NEAR, s.x.max() + NEAR
        y0, y1 = s.y.min() - NEAR, s.y.max() + NEAR
        xs = np.arange(np.floor(x0 / CELL) * CELL, x1, CELL) + CELL / 2
        ys = np.arange(np.floor(y0 / CELL) * CELL, y1, CELL) + CELL / 2
        X, Y = [a.ravel() for a in np.meshgrid(xs, ys)]
        d = cKDTree(s[["x", "y"]].values).query(np.column_stack([X, Y]), k=1)[0]
        k = (d >= MIN_GAP) & (d <= NEAR)
        pts = gpd.GeoDataFrame({"x": X[k], "y": Y[k], "dist_nn_station_m": d[k]},
                               geometry=gpd.points_from_xy(X[k], Y[k]), crs="EPSG:5179")
        poly = gu[gu["code"].str.startswith(prefix)][["name", "geometry"]]
        j = gpd.sjoin(pts, poly, how="inner", predicate="within").drop_duplicates(subset=["x", "y"])
        j["city"], j["gu"] = city, j["name"].str.replace("창원시", "", regex=False)
        rows.append(pd.DataFrame(j[["city", "gu", "x", "y", "dist_nn_station_m"]]))
        log(f"  {city}: 후보 격자 {len(j):,}개")
    c = pd.concat(rows, ignore_index=True)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform(c.x.values, c.y.values)
    c["lon"], c["lat"] = lon, lat
    return c


# ================================================================ 검증
def validate(st, F, has_slope):
    rows = []
    for c in COUNT_COLS + FLOAT_COLS:
        if c == "slope_value" and not has_slope:
            rows.append({"feature": c, "status": "건너뜀(래스터 없음)"})
            continue
        a, b = st[c].to_numpy(float), F[c].to_numpy(float)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        ok = ~np.isnan(a) & ~np.isnan(b)
        diff = np.abs(a[ok] - b[ok])
        tol = 0 if c in COUNT_COLS else (1e-6 * np.maximum(1, np.abs(a[ok])) + 1e-3)
        n_bad = int((diff > tol).sum()) + nan_mismatch
        rows.append({"feature": c, "n": len(a), "n_mismatch": n_bad, "nan_mismatch": nan_mismatch,
                     "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                     "status": "일치" if n_bad == 0 else "불일치"})
    v = pd.DataFrame(rows)
    v.to_csv(OUT_VAL / "validation_v2.csv", index=False, encoding="utf-8-sig")
    return v


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="기존 대여소 재계산 검증만 수행")
    args = ap.parse_args()
    pd.set_option("display.width", 160)

    st = pd.read_csv(DATA / "stations_features_v2.csv", encoding="utf-8-sig", dtype={"station": str})
    raster = slope_raster()
    log(f"경사도 래스터: {raster if raster else '없음 → data/raw/slope/ 에 Slope_All.tif 필요'}")
    if not args.validate and raster is None:
        sys.exit("후보지 피처에는 경사도가 필요합니다. 래스터를 data/raw/slope/ 에 넣고 다시 실행하세요. "
                 "(검증만 하려면 --validate)")

    if args.validate:
        lon, lat, cand = st["lon"].values, st["lat"].values, None
    else:
        log("후보지 격자 생성")
        cand = make_candidates(st)
        lon = np.r_[st["lon"].values, cand["lon"].values]
        lat = np.r_[st["lat"].values, cand["lat"].values]

    log(f"피처 계산 ({len(lon):,}개 지점)")
    F = compute_features(lon, lat, raster)
    Fs = F.iloc[:len(st)].reset_index(drop=True)

    v = validate(st, Fs, raster is not None)
    print("\n=== 기존 대여소 재계산 검증 (v2와 비교) ===")
    print(v.to_string(index=False))
    bad = v[v.status == "불일치"]
    if len(bad):
        sys.exit(f"\n불일치 피처 {len(bad)}개: {bad.feature.tolist()} → 후보지 피처를 만들지 않고 중단합니다. "
                 "이 출력을 그대로 공유해 주세요.")
    log("검증 통과: 조원 v2 피처와 같은 정의로 계산됨")
    if args.validate:
        return

    Fc = F.iloc[len(st):].reset_index(drop=True)
    out = pd.concat([cand.reset_index(drop=True), Fc], axis=1)
    feat_cols = pd.read_csv(DATA / "model_feature_columns_v2.csv", encoding="utf-8-sig")["column"].tolist()
    lead = ["city", "gu", "x", "y", "lat", "lon", "dist_nn_station_m", "GRID_CD"]
    out = out[lead + [c for c in feat_cols if c not in lead]]
    out.to_csv(OUT_CAND, index=False, encoding="utf-8-sig")
    log(f"후보지 피처 저장 → {OUT_CAND} ({len(out):,}행)")
    print(out.groupby("city").size().to_string())


if __name__ == "__main__":
    main()
