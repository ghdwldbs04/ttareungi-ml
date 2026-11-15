# -*- coding: utf-8 -*-
"""
대여소별 주변 환경 피처 만들기 (서울·대전·창원 공통 기준)

    python features.py

입력:  data/processed/<도시>/station_summary.csv  (eda_bike.py plots 결과)
       data/raw/features/*                         (버스, 지하철, 대학, 공원, 자전거도로, 상가, SGIS)
출력:  data/processed/features_station.csv         (대여소 1개 = 1행, 모델 입력)
       outputs/feature_ood.csv                     (대전·창원 값이 서울 범위 밖인 비율)
       outputs/figures/10_feature_compare.png      (도시별 피처 분포 비교)

원천 파일에서 세 도시 주변만 잘라 data/processed/feature_sources/ 에 저장해 두므로 두 번째 실행부터는 빠름.
"""
import io
import re
import sys
import time
import zipfile
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEAT = ROOT / "data" / "raw" / "features"
PROC = ROOT / "data" / "processed"
CACHE = PROC / "feature_sources"
OUT = ROOT / "outputs"
CITIES = {"seoul": "서울 따릉이", "daejeon": "대전 타슈", "changwon": "창원 누비자", "sejong": "세종 어울링"}
RADII = (300, 500)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------
# 좌표: WGS84 위경도 -> UTM-K (EPSG:5179, m). SGIS 격자와 같은 좌표계
# ------------------------------------------------------------
def to_utmk(lat, lon):
    a, f = 6378137.0, 1 / 298.257222101
    k0, lat0, lon0, FE, FN = 0.9996, np.radians(38.0), np.radians(127.5), 1_000_000.0, 2_000_000.0
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))

    def M(p):
        return a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * p
                    - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * p)
                    + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * p)
                    - (35 * e2 ** 3 / 3072) * np.sin(6 * p))

    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2
    A = (lam - lon0) * np.cos(phi)
    x = FE + k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
    y = FN + k0 * (M(phi) - M(lat0) + N * np.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                                           + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


# ------------------------------------------------------------
# 이웃 검색 (scipy 없이): 격자 해시
# ------------------------------------------------------------
class PointIndex:
    def __init__(self, x, y, w=None, cell=500.0):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.w = np.ones(len(self.x)) if w is None else np.asarray(w, float)
        self.cell = cell
        cx, cy = np.floor(self.x / cell).astype(np.int64), np.floor(self.y / cell).astype(np.int64)
        order = np.lexsort((cy, cx))
        self.x, self.y, self.w = self.x[order], self.y[order], self.w[order]
        keys = cx[order] * 10_000_000 + cy[order]
        self.keys, self.starts = np.unique(keys, return_index=True)
        self.ends = np.r_[self.starts[1:], len(keys)]

    def _near(self, px, py, r):
        k = int(np.ceil(r / self.cell))
        cx, cy = int(np.floor(px / self.cell)), int(np.floor(py / self.cell))
        idx = []
        for dx in range(-k, k + 1):
            want = (cx + dx) * 10_000_000 + np.arange(cy - k, cy + k + 1)
            pos = np.searchsorted(self.keys, want)
            pos = np.unique(pos[pos < len(self.keys)])
            for p in pos[np.isin(self.keys[pos], want)]:
                idx.append(np.arange(self.starts[p], self.ends[p]))
        return np.concatenate(idx) if idx else np.array([], int)

    def count_sum(self, px, py, r):
        cnt, sm = np.zeros(len(px)), np.zeros(len(px))
        for i, (a, b) in enumerate(zip(px, py)):
            j = self._near(a, b, r)
            if len(j):
                m = (self.x[j] - a) ** 2 + (self.y[j] - b) ** 2 <= r * r
                cnt[i], sm[i] = m.sum(), self.w[j][m].sum()
        return cnt, sm


def nearest_dist(px, py, tx, ty, block=2000):
    px, py, tx, ty = map(lambda v: np.asarray(v, float), (px, py, tx, ty))
    out = np.empty(len(px))
    for s in range(0, len(px), block):
        d2 = (px[s:s + block, None] - tx[None]) ** 2 + (py[s:s + block, None] - ty[None]) ** 2
        out[s:s + block] = np.sqrt(d2.min(axis=1))
    return out


# ------------------------------------------------------------
# 원천 데이터 읽기 (세 도시 주변 영역만 남겨 캐시)
# ------------------------------------------------------------
def dec(n):
    try:
        return n.encode("cp437").decode("cp949")
    except Exception:
        return n


def read_any(path, **kw):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def find(pattern):
    hits = [p for p in FEAT.glob("*") if re.search(pattern, p.name)]
    if not hits:
        log(f"  !! data/raw/features 에서 '{pattern}' 파일을 못 찾음 -> 이 피처는 건너뜀")
        return None
    return hits[0]


def city_boxes(stations):
    """도시별 대여소 범위 + 2km 여유 (UTM-K)"""
    boxes = {}
    for c, g in stations.groupby("city"):
        boxes[c] = (g.x.min() - 2000, g.x.max() + 2000, g.y.min() - 2000, g.y.max() + 2000)
    return boxes


def clip(df, boxes):
    keep = np.zeros(len(df), bool)
    for x0, x1, y0, y1 in boxes.values():
        keep |= df.x.between(x0, x1).values & df.y.between(y0, y1).values
    return df[keep]


def cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.csv"
    if p.exists():
        return pd.read_csv(p)
    t = time.time()
    df = build()
    if df is None:
        return None
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"  {name}: {len(df):,}개 저장 ({time.time() - t:.0f}초)")
    return df


def pts(df, lat, lon, boxes, keep=()):
    d = df[[lat, lon] + list(keep)].copy()
    d[lat], d[lon] = pd.to_numeric(d[lat], errors="coerce"), pd.to_numeric(d[lon], errors="coerce")
    d = d.dropna(subset=[lat, lon])
    d["x"], d["y"] = to_utmk(d[lat].values, d[lon].values)
    return clip(d.drop(columns=[lat, lon]), boxes)


def src_bus(boxes):
    p = find(r"버스정류장")
    return None if p is None else pts(read_any(p), "위도", "경도", boxes)


def src_subway(boxes):
    p = find(r"도시철도역사")
    if p is None:
        return None
    d = read_any(p)
    d = pts(d, "역위도", "역경도", boxes, keep=["역사명"])
    return d.drop_duplicates("역사명")  # 환승역 중복 제거


def src_univ(boxes):
    p = find(r"대학교.*좌표")
    if p is None:
        return None
    d = read_any(p)
    if "학교상태" in d.columns:
        d = d[~d["학교상태"].astype(str).str.contains("폐")]
    return pts(d, "위도", "경도", boxes, keep=["학교명", "학제"])


def src_park(boxes):
    p = find(r"도시공원")
    if p is None:
        return None
    d = read_any(p)
    d = pts(d, "위도", "경도", boxes, keep=["공원면적"])
    d["공원면적"] = pd.to_numeric(d["공원면적"], errors="coerce").fillna(0)
    return d


def src_bike(boxes):
    p = find(r"자전거도로")
    if p is None:
        return None
    d = read_any(p)
    d["seg"] = np.arange(len(d))
    d["km"] = pd.to_numeric(d["총길이(km)"], errors="coerce").fillna(0)
    a = pts(d, "기점위도", "기점경도", boxes, keep=["seg", "km", "자전거도로종류"])
    b = pts(d, "종점위도", "종점경도", boxes, keep=["seg", "km", "자전거도로종류"])
    return pd.concat([a, b])


def src_shop(boxes):
    p = find(r"상가.*상권")
    if p is None:
        return None
    parts = []
    with zipfile.ZipFile(p) as z:
        for i in z.infolist():
            n = dec(i.filename)
            if not any(k in n for k in ("_서울_", "_대전_", "_경남_", "_세종_")):
                continue
            log("  상가 읽는 중:", n)
            for ch in pd.read_csv(z.open(i.filename), encoding="utf-8", usecols=["상권업종대분류명", "시군구명", "경도", "위도"],
                                  chunksize=300_000, low_memory=False):
                if "_경남_" in n:
                    ch = ch[ch["시군구명"].astype(str).str.contains("창원")]
                parts.append(pts(ch, "위도", "경도", boxes, keep=["상권업종대분류명"]))
    return pd.concat(parts)


def src_sgis(boxes):
    """SGIS 1km 격자: 총인구, 20~39세, 종사자, 사업체"""
    p = find(r"SGIS")
    if p is None:
        return None
    letters = "가나다라마바사아자차카"
    need = set()
    for x0, x1, y0, y1 in boxes.values():
        for xx in range(int(x0 // 100_000), int(x1 // 100_000) + 1):
            for yy in range(int(y0 // 100_000), int(y1 // 100_000) + 1):
                need.add(letters[xx - 7] + letters[yy - 13])
    log("  SGIS 100km 타일:", sorted(need))
    rows = []
    with zipfile.ZipFile(p) as z:
        for i in z.infolist():
            n = dec(i.filename)
            m = re.search(r"2024년_(인구|종사자|사업체)_(..)_1K\.csv$", n)
            if not m or m.group(2) not in need:
                continue
            raw = z.read(i.filename)
            enc = "utf-8-sig"
            try:
                raw.decode(enc)
            except UnicodeDecodeError:
                enc = "cp949"
            d = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype={"격자코드": str})
            d["kind"] = m.group(1)
            rows.append(d)
    d = pd.concat(rows)
    d["통계값"] = pd.to_numeric(d["통계값"], errors="coerce")
    item = d["통계항목"].astype(str)
    young = item.isin([f"in_age_{k:03d}" for k in (5, 6, 7, 8)])
    agesum = item.str.match(r"in_age_0(0[1-9]|1\d|2[01])$")
    out = pd.DataFrame({
        "pop": d[item == "to_in_001"].groupby("격자코드")["통계값"].sum(),
        "pop_age": d[agesum].groupby("격자코드")["통계값"].sum(),
        "young": d[young].groupby("격자코드")["통계값"].sum(),
        "workers": d[(d.kind == "종사자") & item.str.startswith("to_em")].groupby("격자코드")["통계값"].sum(),
        "firms": d[(d.kind == "사업체") & item.str.startswith("to_fa")].groupby("격자코드")["통계값"].sum(),
    }).reset_index().rename(columns={"index": "격자코드"})
    out["pop"] = out["pop"].fillna(out["pop_age"])
    return out.drop(columns="pop_age")


def grid_code(x, y):
    letters = "가나다라마바사아자차카"
    xi, yi = (np.asarray(x) // 1000).astype(int), (np.asarray(y) // 1000).astype(int)
    return [letters[a // 100 - 7] + letters[b // 100 - 13] + f"{a % 100:02d}{b % 100:02d}" for a, b in zip(xi, yi)]


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def load_stations():
    rows = []
    for c in CITIES:
        p = PROC / c / "station_summary.csv"
        if not p.exists():
            log(f"  {c}: station_summary.csv 없음 (eda_bike.py plots 먼저)")
            continue
        g = pd.read_csv(p, dtype={"station": str})
        n0 = len(g)
        g = g.dropna(subset=["lat", "lon"])
        log(f"  {c}: 대여소 {n0:,}개 중 좌표 있는 {len(g):,}개 사용")
        g["city"] = c
        rows.append(g)
    s = pd.concat(rows, ignore_index=True)
    s["x"], s["y"] = to_utmk(s.lat.values, s.lon.values)
    return s


def build():
    log("대여소 불러오기")
    s = load_stations()
    boxes = city_boxes(s)
    px, py = s.x.values, s.y.values
    F = pd.DataFrame(index=s.index)

    log("원천 데이터 준비 (처음 한 번만 오래 걸림)")
    src = {k: cached(k, lambda f=f: f(boxes)) for k, f in
           [("bus", src_bus), ("subway", src_subway), ("univ", src_univ), ("park", src_park),
            ("bike", src_bike), ("shop", src_shop), ("sgis", src_sgis)]}

    log("피처 계산")
    if src["bus"] is not None:
        idx = PointIndex(src["bus"].x, src["bus"].y)
        for r in RADII:
            F[f"bus_{r}"] = idx.count_sum(px, py, r)[0]
    if src["subway"] is not None:
        # 창원은 도시철도가 없어 가장 가까운 역이 수십~수백 km -> 5km로 상한 (5km = 사실상 '역 없음')
        F["dist_subway_m"] = np.minimum(nearest_dist(px, py, src["subway"].x, src["subway"].y), 5000)
        F["subway_1000"] = PointIndex(src["subway"].x, src["subway"].y, cell=1000).count_sum(px, py, 1000)[0]
    if src["univ"] is not None:
        F["dist_univ_m"] = nearest_dist(px, py, src["univ"].x, src["univ"].y)
    if src["park"] is not None:
        F["dist_park_m"] = nearest_dist(px, py, src["park"].x, src["park"].y)
        n, a = PointIndex(src["park"].x, src["park"].y, src["park"]["공원면적"]).count_sum(px, py, 500)
        F["park_500"], F["park_area_500"] = n, a
    if src["bike"] is not None:
        b = src["bike"]
        F["bike_pts_500"] = PointIndex(b.x, b.y).count_sum(px, py, 500)[0]
        # 반경 500m 안에 시점·종점이 걸치는 구간의 길이 합 (구간마다 한 번만)
        seg_km = b.groupby("seg")["km"].first()
        idx = PointIndex(b.x, b.y, b.seg)
        tot = np.zeros(len(px))
        for i, (a_, c_) in enumerate(zip(px, py)):
            j = idx._near(a_, c_, 500)
            if len(j):
                m = (idx.x[j] - a_) ** 2 + (idx.y[j] - c_) ** 2 <= 250_000
                segs = np.unique(idx.w[j][m]).astype(int)
                tot[i] = seg_km.reindex(segs).sum()
        F["bike_km_500"] = tot
    if src["shop"] is not None:
        sh = src["shop"]
        idx = PointIndex(sh.x, sh.y)
        food = sh[sh["상권업종대분류명"].astype(str).str.contains("음식")]
        idf = PointIndex(food.x, food.y)
        for r in RADII:
            F[f"shop_{r}"] = idx.count_sum(px, py, r)[0]
        F["food_500"] = idf.count_sum(px, py, 500)[0]
    if src["sgis"] is not None:
        g = src["sgis"].set_index("격자코드")
        codes = grid_code(px, py)
        F["pop_1km"] = g["pop"].reindex(codes).fillna(0).values
        F["young_share_1km"] = (g["young"].reindex(codes).values / F["pop_1km"].replace(0, np.nan)).clip(0, 1)
        F["workers_1km"] = g["workers"].reindex(codes).fillna(0).values
        F["firms_1km"] = g["firms"].reindex(codes).fillna(0).values
    # 대여소 네트워크 (같은 도시 안)
    F["stations_500"], F["dist_nn_station_m"] = 0.0, np.nan
    for c, g in s.groupby("city"):
        ii = g.index.values
        F.loc[ii, "stations_500"] = PointIndex(g.x, g.y).count_sum(g.x.values, g.y.values, 500)[0] - 1
        gx, gy = g.x.values, g.y.values
        d2 = (gx[:, None] - gx[None]) ** 2 + (gy[:, None] - gy[None]) ** 2
        np.fill_diagonal(d2, np.inf)
        F.loc[ii, "dist_nn_station_m"] = np.sqrt(d2.min(axis=1))

    out = pd.concat([s[["city", "station", "lat", "lon", "x", "y", "daily_mean", "total", "active_days"]], F], axis=1)
    out["log_y"] = np.log1p(out["daily_mean"])
    out["rank_pct"] = out.groupby("city")["daily_mean"].rank(pct=True)
    out.to_csv(PROC / "features_station.csv", index=False, encoding="utf-8-sig")
    log(f"저장: {PROC / 'features_station.csv'}  ({len(out):,}행 × {F.shape[1]}개 피처)")
    return out, list(F.columns)


def report(out, feats):
    # 서울 범위(1~99%) 밖 비율
    sl = out[out.city == "seoul"]
    rows = []
    for f in feats:
        lo, hi = sl[f].quantile(0.01), sl[f].quantile(0.99)
        r = {"feature": f, "seoul_p01": round(lo, 3), "seoul_p99": round(hi, 3)}
        for c in CITIES:
            v = out.loc[out.city == c, f]
            r[f"{c}_median"] = round(float(v.median()), 3) if len(v) else None
            if c != "seoul" and len(v):
                r[f"{c}_out_of_seoul_range_%"] = round(float(((v < lo) | (v > hi)).mean() * 100), 1)
        corr = out[out.city == "seoul"][[f, "log_y"]].corr(method="spearman").iloc[0, 1]
        r["seoul_spearman_with_log_y"] = round(float(corr), 3)
        rows.append(r)
    rep = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    rep.to_csv(OUT / "feature_ood.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(rep.to_string(index=False))

    # 그림: 도시별 피처 분포 (log1p 스케일 박스플롯)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"seoul": "#2a6fdb", "daejeon": "#e07b39", "changwon": "#3a9e6b", "sejong": "#8e5bd0"}
    n = len(feats)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(14, 2.6 * np.ceil(n / cols)))
    for ax, f in zip(axes.flat, feats):
        data = [np.log1p(out.loc[out.city == c, f].dropna().clip(lower=0)) for c in CITIES]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], CITIES):
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.5)
        ax.set_xticks(list(range(1, len(CITIES) + 1)), [v.split()[0] for v in CITIES.values()])
        ax.set_title(f, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("대여소 주변 환경 — 도시별 분포 (log1p 스케일)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "10_feature_compare.png", dpi=200, bbox_inches="tight", facecolor="white")
    log("그림 저장: outputs/figures/10_feature_compare.png")


if __name__ == "__main__":
    out, feats = build()
    report(out, feats)
