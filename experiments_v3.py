# -*- coding: utf-8 -*-
"""
공간통계 보완 분석 — "지역마다 효과가 어떻게 다른가"를 모형으로 보여주기

    pip install mgwr                 # 처음 한 번 (없으면 GWR은 내장 구현으로 돌아감)
    python experiments_v3.py         # S1~S4 전부 (로컬 PC 기준 5~15분)
    python experiments_v3.py --mgwr  # 서울 MGWR(변수별 공간 스케일)까지. 30~90분 걸릴 수 있음

필요: model.py, viz_map.py, features.py, features_extra.py, grid_features.py 와 같은 폴더
      data/processed/features_station_v2.csv  (+ grid_candidates.csv 없으면 자동 생성)
출력: outputs/model_v3/*.csv, summary.txt / outputs/figures/model_v3/*.png

S1  Moran's I + LISA : 이용량·예측 오차가 공간적으로 뭉쳐 있는가 (KNN 8, 999회 순열)
S2  GWR (세 도시)     : 변수 효과(계수)가 도시 안에서 지역마다 어떻게 달라지는가 + 서울 계수 지도
    MGWR (서울, 옵션) : 변수마다 영향 범위(공간 스케일)가 다른가
S3  도시 상호작용 회귀 : 도시 간 차이를 '수준(절편)'과 '효과(기울기)'로 나눠 검정 (세 도시라 고정효과형 다수준 구조)
S4  후보지 수요 지도   : 서울+현지 모델로 대여소가 없는 250m 격자의 예상 수요 등급
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import PathPatch, Patch
from scipy.spatial import cKDTree
from scipy.stats import chi2, t as tdist, norm as znorm

from model import gbm, font, LABELS, BASE_FEATS, EXTRA, SHARES, CITIES, SEED

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "model_v3"
FIG = ROOT / "outputs" / "figures" / "model_v3"
DO_MGWR = "--mgwr" in sys.argv
QUICK = "--quick" in sys.argv
PERMS = 99 if QUICK else 999
K_NN = 8
BLOCK = 2000
FULL = BASE_FEATS + EXTRA
SAFE = [f for f in FULL if f not in ("cycleway_m_500", "dist_cycleway_m")]
FEATS_OF = {"seoul": FULL, "daejeon": FULL, "changwon": SAFE}
# GWR 변수: 서울 SHAP·중요도 상위 + 선행연구 핵심 (다중공선성 줄이려고 8개로 제한)
VARS_GWR = ["shop_300", "workers_1km", "apt_share_1km", "cycleway_m_500", "dist_subway_m", "dist_river_m",
            "dist_nn_station_m", "bus_500"]
# 도시 비교 회귀 변수: 세 도시에서 같은 의미로 측정되는 것만 (지하철·자전거도로 제외)
VARS_X = ["shop_300", "workers_1km", "pop_1km", "apt_share_1km", "bus_500", "dist_river_m", "dist_univ_m",
          "dist_nn_station_m"]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def X_of(d, feats):
    X = d[feats].astype(float).copy()
    return X.fillna(X.median())


def tf(d, vars_):
    """개수·거리 → log1p, 비율 → 그대로"""
    X = d[vars_].astype(float).copy()
    for v in vars_:
        if v not in SHARES:
            X[v] = np.log1p(X[v].clip(lower=0))
    return X.fillna(X.median())


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say("  그림:", FIG / name)


def to_csv(d, name, index=False):
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / name, index=index, encoding="utf-8-sig")
    say("  표:", OUT / name)


def base_map(ax, lst, gray=True):
    from viz_map import draw_gu, outline, extent, frame
    if gray:
        for g in lst:
            ax.add_patch(PathPatch(g["path"], facecolor="#f4f4f4", edgecolor="none", zorder=0))
    outline(ax, lst)
    draw_gu(ax, lst, fs=7)
    frame(ax, extent(lst))


# ------------------------------------------------------------
def load():
    df = pd.read_csv(PROC / "features_station_v2.csv", dtype={"station": str})
    from viz_map import load_gu, assign_gu
    gus = load_gu()
    df = assign_gu(df, gus)
    df["gu"] = df["gu"].fillna("기타")
    df["block"] = df.city + "_" + (df.x // BLOCK).astype(int).astype(str) + "_" + (df.y // BLOCK).astype(int).astype(str)
    return df, gus


def residuals(df):
    """서울: 자치구 CV 예측 오차 / 대전·창원: 서울 모델 제로샷 오차 (최종 피처 설정)"""
    from sklearn.model_selection import GroupKFold
    S = df[df.city == "seoul"].reset_index(drop=True)
    p = np.zeros(len(S))
    for tr, te in GroupKFold(5).split(S, S.log_y, S.gu):
        p[te] = gbm().fit(X_of(S.iloc[tr], FULL), S.log_y.iloc[tr]).predict(X_of(S.iloc[te], FULL))
    df = df.copy()
    df["resid"] = np.nan
    df.loc[df.city == "seoul", "resid"] = p - S.log_y.values
    for c in ("daejeon", "changwon"):
        T = df[df.city == c]
        m = gbm().fit(X_of(S, FEATS_OF[c]), S.log_y)
        df.loc[df.city == c, "resid"] = m.predict(X_of(T, FEATS_OF[c])) - T.log_y.values
    return df


# ------------------------------------------------------------ S1 Moran / LISA
def knn(x, y, k=K_NN):
    return cKDTree(np.c_[x, y]).query(np.c_[x, y], k=k + 1)[1][:, 1:]


def moran(v, nb, perms=PERMS, seed=SEED):
    z = v - v.mean()
    I = float((z * z[nb].mean(1)).sum() / (z * z).sum())
    rng = np.random.default_rng(seed)
    sims = np.empty(perms)
    for i in range(perms):
        zp = rng.permutation(z)
        sims[i] = (zp * zp[nb].mean(1)).sum() / (zp * zp).sum()
    p = (np.sum(sims >= I) + 1) / (perms + 1)
    return I, p, (I - sims.mean()) / sims.std()


def lisa(v, nb, perms=PERMS, seed=SEED):
    """조건부 순열(이웃을 무작위 k개로 바꿔 끼움) 근사 — 유의한 HH·LL·HL·LH 구분"""
    z = (v - v.mean()) / v.std()
    n, k = nb.shape
    lag = z[nb].mean(1)
    Ii = z * lag
    rng = np.random.default_rng(seed)
    ge = np.zeros(n)
    for s in range(0, perms, 50):
        m = min(50, perms - s)
        r = rng.integers(0, n, size=(n, m, k))
        sim = z[:, None] * z[r].mean(2)
        ge += np.where(Ii[:, None] >= 0, sim >= Ii[:, None], sim <= Ii[:, None]).sum(1)
    p = (ge + 1) / (perms + 1)
    q = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    return np.where(p < 0.05, q, "ns"), p


def s1_moran(df, gus):
    say(f"\n[S1] Moran's I (KNN {K_NN}, {PERMS}회 순열)")
    rows = []
    cats = {}
    for c in CITIES:
        T = df[df.city == c].reset_index(drop=True)
        nb = knn(T.x.values, T.y.values)
        for col, lab in (("log_y", "이용량 log(1+하루 대여)"), ("resid", "예측 오차 (예측−실제)")):
            I, p, z = moran(T[col].values, nb)
            rows.append({"도시": CITIES[c], "변수": lab, "Moran's I": round(I, 3), "z": round(z, 1), "p": round(p, 4)})
        cats[c], _ = lisa(T.log_y.values, nb)
        T["lisa"] = cats[c]
        cnt = pd.Series(cats[c]).value_counts()
        rows.append({"도시": CITIES[c], "변수": "LISA 이용량 군집 대여소 수",
                     "Moran's I": None, "z": None, "p": None,
                     **{k: int(cnt.get(k, 0)) for k in ("HH", "LL", "HL", "LH", "ns")}})
        T[["city", "station", "gu", "x", "y", "log_y", "lisa"]].to_csv(OUT / f"S1_lisa_{c}.csv", index=False, encoding="utf-8-sig")
    t = pd.DataFrame(rows)
    to_csv(t, "S1_moran.csv")
    say(t.to_string(index=False))

    col = {"HH": "#c0392b", "LL": "#2a6fdb", "HL": "#f5a59a", "LH": "#9cc3f0", "ns": "#d9d9d9"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), gridspec_kw={"width_ratios": [1.25, 0.9, 1.1]})
    for ax, c in zip(axes, CITIES):
        T = df[df.city == c].reset_index(drop=True)
        T["lisa"] = cats[c]
        base_map(ax, gus[c])
        for k in ("ns", "LH", "HL", "LL", "HH"):
            d = T[T.lisa == k]
            ax.scatter(d.x, d.y, s=9 if k != "ns" else 5, color=col[k], linewidths=0, zorder=4)
        I = t[(t["도시"] == CITIES[c]) & t["변수"].str.startswith("이용량")]["Moran's I"].iloc[0]
        ax.set_title(f"{CITIES[c]}  (Moran's I = {I:.2f})", loc="left", fontsize=13)
    axes[2].legend(handles=[Patch(color=col["HH"], label="HH: 많이 타는 곳끼리 모임"),
                            Patch(color=col["LL"], label="LL: 적게 타는 곳끼리 모임"),
                            Patch(color=col["HL"], label="HL: 주변보다 혼자 높음"),
                            Patch(color=col["LH"], label="LH: 주변보다 혼자 낮음"),
                            Patch(color=col["ns"], label="유의하지 않음 (p ≥ 0.05)")],
                   frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("이용량의 공간 군집 (LISA) — 대여소 이용은 무작위가 아니라 지역 단위로 뭉쳐 있음", x=0.01, ha="left", fontsize=15)
    save(fig, "S1_lisa_map.png")
    return t


# ------------------------------------------------------------ S2 GWR
def design(T, vars_):
    X = tf(T, vars_)
    X = X.loc[:, X.std() > 1e-9]
    Xs = (X - X.mean()) / X.std()
    y = T.log_y.values
    ys = (y - y.mean()) / y.std()
    return Xs, ys


def ols(X, y):
    A = np.c_[np.ones(len(X)), X]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    n, k = A.shape
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    sig = np.sqrt(e @ e / n)
    aicc = 2 * n * np.log(sig) + n * np.log(2 * np.pi) + n * (n + k) / (n - 2 - k)
    return b, se, e, 1 - e @ e / ((y - y.mean()) @ (y - y.mean())), aicc


class MiniGWR:
    """mgwr이 없을 때 쓰는 GWR (적응형 bisquare, AICc 황금분할 탐색, Fotheringham·da Silva 보정 t)"""

    def __init__(self, coords, y, X):
        self.c, self.y = np.asarray(coords, float), np.asarray(y, float).ravel()
        self.A = np.c_[np.ones(len(self.y)), np.asarray(X, float)]
        self.tree = cKDTree(self.c)

    def fit(self, bw, full=True):
        n, k = self.A.shape
        bw = int(bw)
        d, nb = self.tree.query(self.c, k=min(bw + 1, n))
        h = d[:, -1:] + 1e-9
        W = (1 - (d / h) ** 2) ** 2
        B, S_ii, CC = np.zeros((n, k)), np.zeros(n), np.zeros((n, k))
        yhat = np.zeros(n)
        for i in range(n):
            Ai, wi, yi = self.A[nb[i]], W[i], self.y[nb[i]]
            M = np.linalg.pinv((Ai * wi[:, None]).T @ Ai)
            C = M @ (Ai * wi[:, None]).T
            B[i] = C @ yi
            yhat[i] = self.A[i] @ B[i]
            S_ii[i] = self.A[i] @ C[:, 0]
            if full:
                CC[i] = (C ** 2).sum(1)
        e = self.y - yhat
        trS = S_ii.sum()
        sig = np.sqrt(e @ e / n)
        aicc = 2 * n * np.log(sig) + n * np.log(2 * np.pi) + n * (n + trS) / (n - 2 - trS)
        res = {"params": B, "resid": e, "trS": trS, "aicc": aicc, "bw": bw,
               "R2": 1 - e @ e / ((self.y - self.y.mean()) @ (self.y - self.y.mean()))}
        if full:
            s2 = e @ e / (n - trS)
            se = np.sqrt(s2 * CC)
            tv = B / se
            alpha = 0.05 / (trS / k)
            tcrit = tdist.ppf(1 - alpha / 2, n - trS)
            res.update(se=se, t=tv, tcrit=tcrit, sig=np.abs(tv) > tcrit)
        return res

    def search(self):
        n, k = self.A.shape
        lo, hi = k + 10, n - 1
        f = lambda b: self.fit(b, full=False)["aicc"]
        g = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c1, c2 = int(round(b - g * (b - a))), int(round(a + g * (b - a)))
        f1, f2 = f(c1), f(c2)
        while b - a > 4:
            if f1 < f2:
                b, c2, f2 = c2, c1, f1
                c1 = int(round(b - g * (b - a)))
                f1 = f(c1)
            else:
                a, c1, f1 = c1, c2, f2
                c2 = int(round(a + g * (b - a)))
                f2 = f(c2)
        return c1 if f1 < f2 else c2


def run_gwr(T, vars_):
    Xs, ys = design(T, vars_)
    coords = T[["x", "y"]].values
    try:
        from mgwr.sel_bw import Sel_BW
        from mgwr.gwr import GWR
        bw = Sel_BW(coords, ys.reshape(-1, 1), Xs.values).search()
        r = GWR(coords, ys.reshape(-1, 1), Xs.values, bw).fit()
        sig = np.asarray(r.filter_tvals()) != 0
        out = {"params": np.asarray(r.params), "sig": sig, "resid": np.asarray(r.resid_response).ravel(),
               "aicc": float(r.aicc), "R2": float(r.R2), "bw": bw, "trS": float(r.tr_S), "engine": "mgwr"}
    except ImportError:
        g = MiniGWR(coords, ys, Xs.values)
        bw = g.search()
        out = g.fit(bw)
        out["engine"] = "내장 GWR"
    return Xs, ys, out


def s2_gwr(df, gus):
    say("\n[S2] GWR — 변수 효과가 도시 안에서 지역마다 다른가")
    rows, comp, keep = [], [], {}
    for c in CITIES:
        T = df[df.city == c].reset_index(drop=True)
        vars_ = [v for v in VARS_GWR if not (c == "changwon" and v == "cycleway_m_500")]  # 창원 OSM 자전거도로는 데이터 공백
        Xs, ys, r = run_gwr(T, vars_)
        b, se, e_ols, r2_ols, aicc_ols = ols(Xs.values, ys)
        nb = knn(T.x.values, T.y.values)
        I_ols, p_ols, _ = moran(e_ols, nb)
        I_gwr, p_gwr, _ = moran(r["resid"], nb)
        comp.append({"도시": CITIES[c], "엔진": r["engine"], "대여소 수": len(T), "대역폭(이웃 수)": int(r["bw"]),
                     "OLS R²": round(r2_ols, 3), "GWR R²": round(r["R2"], 3), "OLS AICc": round(aicc_ols, 1),
                     "GWR AICc": round(r["aicc"], 1), "잔차 Moran's I (OLS)": round(I_ols, 3),
                     "잔차 Moran's I (GWR)": round(I_gwr, 3), "잔차 Moran p (GWR)": round(p_gwr, 3)})
        P, S = r["params"], r["sig"]
        for j, v in enumerate(["(절편)"] + list(Xs.columns)):
            pj, sj = P[:, j], S[:, j]
            rows.append({"도시": CITIES[c], "변수": LABELS.get(v, v), "OLS 계수": round(b[j], 3),
                         "OLS p": round(2 * (1 - tdist.cdf(abs(b[j] / se[j]), len(T) - len(b))), 4),
                         "GWR 중앙값": round(np.median(pj), 3), "GWR 25%": round(np.percentile(pj, 25), 3),
                         "GWR 75%": round(np.percentile(pj, 75), 3), "GWR 최소": round(pj.min(), 3), "GWR 최대": round(pj.max(), 3),
                         "유의(+) 대여소 %": round(100 * np.mean(sj & (pj > 0)), 1),
                         "유의(−) 대여소 %": round(100 * np.mean(sj & (pj < 0)), 1)})
        keep[c] = (T, Xs, r)
        say(f"  {CITIES[c]}: 대역폭 {int(r['bw'])}, R² OLS {r2_ols:.3f} → GWR {r['R2']:.3f}, "
            f"잔차 Moran's I {I_ols:.3f} → {I_gwr:.3f} ({r['engine']})")
    t1, t2 = pd.DataFrame(comp), pd.DataFrame(rows)
    to_csv(t1, "S2_gwr_fit.csv")
    to_csv(t2, "S2_gwr_coefficients.csv")
    say(t2[t2["변수"] != "(절편)"][["도시", "변수", "OLS 계수", "GWR 중앙값", "유의(+) 대여소 %", "유의(−) 대여소 %"]].to_string(index=False))

    # 서울 계수 지도
    T, Xs, r = keep["seoul"]
    cols = list(Xs.columns)
    fig, axes = plt.subplots(2, 4, figsize=(24, 11.5))
    for ax, v in zip(axes.ravel(), cols):
        j = cols.index(v) + 1
        pj, sj = r["params"][:, j], r["sig"][:, j]
        lim = max(np.percentile(np.abs(pj), 95), 0.05)
        base_map(ax, gus["seoul"])
        ax.scatter(T.x[~sj], T.y[~sj], s=4, color="#cfcfcf", linewidths=0, zorder=4)
        sc = ax.scatter(T.x[sj], T.y[sj], c=pj[sj], cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), s=8, linewidths=0, zorder=5)
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
        ax.set_title(f"{LABELS.get(v, v)}\n유의 대여소 {100 * sj.mean():.0f}% (+{100 * np.mean(sj & (pj > 0)):.0f}% / −{100 * np.mean(sj & (pj < 0)):.0f}%)",
                     loc="left", fontsize=11.5)
    fig.suptitle("서울 GWR 지역별 계수 — 빨강: 값이 클수록 이용 많음, 파랑: 적음, 회색: 유의하지 않음 (표준화 계수)",
                 x=0.01, ha="left", fontsize=15)
    save(fig, "S2_gwr_seoul_coef_maps.png")

    # 세 도시 계수 분포 비교
    show = [v for v in VARS_GWR if v not in ("cycleway_m_500", "dist_subway_m")]
    fig, axes = plt.subplots(1, len(show), figsize=(3.1 * len(show), 4.6), sharey=False)
    colors = {"seoul": "#2a6fdb", "daejeon": "#e07b39", "changwon": "#3a9e6b"}
    for ax, v in zip(axes, show):
        data, labs, cs = [], [], []
        for c in CITIES:
            T, Xs, r = keep[c]
            if v in Xs.columns:
                data.append(r["params"][:, list(Xs.columns).index(v) + 1])
                labs.append(CITIES[c].split()[0])
                cs.append(colors[c])
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        ax.set_xticks(range(1, len(labs) + 1), labs)
        for patch, col in zip(bp["boxes"], cs):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(LABELS.get(v, v), fontsize=11)
    axes[0].set_ylabel("GWR 지역 계수 (표준화)")
    fig.suptitle("같은 변수라도 도시 안·도시 간 효과 크기가 다름 — GWR 지역 계수 분포 (상자: 25~75%)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, "S2_gwr_coef_by_city.png")
    return t1, t2


def s2_mgwr(df):
    say("\n[S2b] 서울 MGWR — 변수마다 영향 범위(공간 스케일)")
    try:
        from mgwr.sel_bw import Sel_BW
        from mgwr.gwr import MGWR
    except ImportError:
        say("  mgwr 미설치 → 생략 (pip install mgwr)")
        return None
    T = df[df.city == "seoul"].reset_index(drop=True)
    Xs, ys = design(T, VARS_GWR)
    coords = T[["x", "y"]].values
    sel = Sel_BW(coords, ys.reshape(-1, 1), Xs.values, multi=True)
    sel.search(multi_bw_min=[30], verbose=True)
    r = MGWR(coords, ys.reshape(-1, 1), Xs.values, sel).fit()
    bws = np.asarray(sel.bw[0]).ravel()
    P = np.asarray(r.params)
    S = np.asarray(r.filter_tvals()) != 0
    t = pd.DataFrame({"변수": ["(절편)"] + [LABELS.get(v, v) for v in Xs.columns], "대역폭(이웃 수)": bws.astype(int),
                      "대역폭/전체(%)": (bws / len(T) * 100).round(1), "계수 중앙값": np.median(P, 0).round(3),
                      "유의 대여소 %": (S.mean(0) * 100).round(1)})
    to_csv(t, "S2b_mgwr_seoul.csv")
    say(f"  MGWR R² {float(r.R2):.3f}, AICc {float(r.aicc):.1f}")
    say(t.to_string(index=False))
    return t


# ------------------------------------------------------------ S3 도시 상호작용 회귀
def ols_cluster(y, A, groups):
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    n, k = A.shape
    bread = np.linalg.pinv(A.T @ A)
    meat = np.zeros((k, k))
    gs = pd.Series(groups)
    for _, ix in gs.groupby(gs).groups.items():
        s = A[ix].T @ e[ix]
        meat += np.outer(s, s)
    G = gs.nunique()
    V = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
    rss = e @ e
    r2 = 1 - rss / ((y - y.mean()) @ (y - y.mean()))
    return {"b": b, "V": V, "r2": r2, "adj": 1 - (1 - r2) * (n - 1) / (n - k), "aic": n * np.log(rss / n) + 2 * k, "k": k}


def s3_city_regression(df):
    say("\n[S3] 도시 상호작용 회귀 — 도시 간 차이 = 수준(절편) 차이인가, 효과(기울기) 차이인가")
    d = df.reset_index(drop=True)
    X = tf(d, VARS_X)
    X = (X - X.mean()) / X.std()
    y = d.log_y.values
    dj, cw = (d.city == "daejeon").astype(float).values, (d.city == "changwon").astype(float).values
    groups = np.where(d.city == "seoul", d.city + "_" + d.gu, d.block)
    one = np.ones(len(d))
    names_x = list(X.columns)
    M = {"M1 도시만": np.c_[one, dj, cw],
         "M2 입지만": np.c_[one, X.values],
         "M3 도시 + 입지": np.c_[one, dj, cw, X.values],
         "M4 도시 + 입지 + 도시×입지": np.c_[one, dj, cw, X.values, X.values * dj[:, None], X.values * cw[:, None]]}
    fits = {k: ols_cluster(y, A, groups) for k, A in M.items()}
    k_x = len(names_x)
    f4 = fits["M4 도시 + 입지 + 도시×입지"]
    ii = np.arange(3 + k_x, 3 + 3 * k_x)
    bI, VI = f4["b"][ii], f4["V"][np.ix_(ii, ii)]
    W = float(bI @ np.linalg.pinv(VI) @ bI)
    p_slope = 1 - chi2.cdf(W, len(ii))
    f3 = fits["M3 도시 + 입지"]
    lv = f3["b"][1:3]
    lvse = np.sqrt(np.diag(f3["V"])[1:3])
    W_lv = float(lv @ np.linalg.pinv(f3["V"][1:3, 1:3]) @ lv)
    t_fit = pd.DataFrame([{"모형": k, "R²": round(f["r2"], 3), "조정 R²": round(f["adj"], 3), "AIC": round(f["aic"], 1),
                           "모수 수": f["k"]} for k, f in fits.items()])
    t_fit = pd.concat([t_fit, pd.DataFrame([
        {"모형": "검정: 도시별 수준 차이 = 0 (M3, 군집강건 Wald)", "R²": None, "AIC": round(W_lv, 1), "모수 수": 2,
         "조정 R²": f"{1 - chi2.cdf(W_lv, 2):.2e}"},
        {"모형": "검정: 도시별 기울기 차이 = 0 (M4, 군집강건 Wald)", "R²": None, "AIC": round(W, 1), "모수 수": len(ii),
         "조정 R²": f"{p_slope:.2e}"}])], ignore_index=True)
    t_fit = t_fit.rename(columns={"AIC": "AIC / Wald χ²", "조정 R²": "조정 R² / p값"})
    to_csv(t_fit, "S3_fit_and_tests.csv")
    say(t_fit.to_string(index=False))
    t_lv = pd.DataFrame({"도시": ["대전 타슈", "창원 누비자"], "같은 입지에서 log 차이 (서울 기준)": lv.round(3),
                         "95% CI 하한": (lv - 1.96 * lvse).round(3), "95% CI 상한": (lv + 1.96 * lvse).round(3),
                         "배율 exp(차이)": np.exp(lv).round(2)})
    to_csv(t_lv, "S3_level_difference.csv")
    say(t_lv.to_string(index=False))

    rows = []
    b, V = f4["b"], f4["V"]
    for j, v in enumerate(names_x):
        bx = 3 + j
        for c, extra in (("서울 따릉이", None), ("대전 타슈", 3 + k_x + j), ("창원 누비자", 3 + 2 * k_x + j)):
            if extra is None:
                est, var = b[bx], V[bx, bx]
            else:
                est, var = b[bx] + b[extra], V[bx, bx] + V[extra, extra] + 2 * V[bx, extra]
            se = np.sqrt(var)
            row = {"변수": LABELS.get(v, v), "도시": c, "기울기": est, "하한": est - 1.96 * se, "상한": est + 1.96 * se}
            if extra is not None:
                row["서울과 차이 p"] = 2 * (1 - znorm.cdf(abs(b[extra]) / np.sqrt(V[extra, extra])))
            rows.append(row)
    t_sl = pd.DataFrame(rows).round(3)
    to_csv(t_sl, "S3_slopes_by_city.csv")
    say(t_sl.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 0.75 * k_x + 1.5))
    colors = {"서울 따릉이": "#2a6fdb", "대전 타슈": "#e07b39", "창원 누비자": "#3a9e6b"}
    for i, v in enumerate(names_x):
        for j, c in enumerate(colors):
            r = t_sl[(t_sl["변수"] == LABELS.get(v, v)) & (t_sl["도시"] == c)].iloc[0]
            yy = i + (j - 1) * 0.22
            ax.plot([r["하한"], r["상한"]], [yy, yy], color=colors[c], lw=2)
            ax.plot(r["기울기"], yy, "o", color=colors[c], ms=6, label=c if i == 0 else None)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_yticks(range(k_x), [LABELS.get(v, v) for v in names_x])
    ax.invert_yaxis()
    ax.set_xlabel("표준화 기울기 (95% 신뢰구간, 군집강건 표준오차)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"도시별 입지 효과 — 기울기 차이 검정 p = {p_slope:.3g}, 수준 차이: 대전 ×{np.exp(lv[0]):.2f}, 창원 ×{np.exp(lv[1]):.2f}",
                 loc="left", fontsize=11.5)
    save(fig, "S3_slopes_forest.png")
    return t_fit, t_lv, t_sl


# ------------------------------------------------------------ S4 후보지 수요 지도
def s4_candidates(df, gus):
    say("\n[S4] 후보지 수요 지도 (대여소 없는 250m 격자, 서울+현지 모델)")
    p = PROC / "grid_candidates.csv"
    if not p.exists():
        import grid_features
        grid_features.build()
    G = pd.read_csv(p)
    S = df[df.city == "seoul"]
    rows, tops = [], []
    cmap = ListedColormap(["#a50f15", "#ef6548", "#fdbb84", "#c6dbef", "#6b8fb3"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    for ax, c in zip(axes, ("daejeon", "changwon")):
        feats = FEATS_OF[c]
        T = df[df.city == c]
        pool = pd.concat([S, T])
        Xp = X_of(pool, feats).assign(is_local=(pool.city == c).astype(float).values)
        m = gbm().fit(Xp, pool.log_y, sample_weight=np.where(pool.city == c, 5.0, 1.0))
        g = G[G.city == c].copy()
        g["pred_log"] = m.predict(X_of(g, feats).assign(is_local=1.0))
        g["pred_daily"] = np.expm1(g.pred_log).clip(lower=0)
        q = np.quantile(T.log_y, [0.2, 0.4, 0.6, 0.8])
        g["등급"] = 5 - np.searchsorted(q, g.pred_log)
        cnt = g["등급"].value_counts().sort_index()
        rows.append({"도시": CITIES[c], "후보 격자 수": len(g), **{f"{k}등급": int(cnt.get(k, 0)) for k in range(1, 6)},
                     "기존 대여소 상위 20% 기준 (하루 대여)": round(float(np.expm1(q[-1])), 1)})
        st = T[["station", "x", "y"]].reset_index(drop=True)
        dnn, inn = cKDTree(st[["x", "y"]].values).query(g[["x", "y"]].values)
        g["가까운 기존 대여소"], g["거리(m)"] = st.station.values[inn], dnn.round(0)
        top = g.sort_values("pred_log", ascending=False).head(20)
        tops.append(top[["city", "gu", "x", "y", "pred_daily", "등급", "가까운 기존 대여소", "거리(m)"]].round(1))
        g.to_csv(OUT / f"S4_candidates_{c}.csv", index=False, encoding="utf-8-sig")
        base_map(ax, gus[c])
        for k in (5, 4, 3, 2, 1):
            d = g[g["등급"] == k]
            ax.scatter(d.x, d.y, c=d["등급"], cmap=cmap, norm=norm, s=14, marker="s", linewidths=0, zorder=4)
        ax.scatter(T.x, T.y, s=3, color="black", linewidths=0, zorder=5, alpha=0.6)
        ax.set_title(f"{CITIES[c]} — 후보 격자 {len(g):,}개 중 1등급 {int(cnt.get(1, 0))}개", loc="left", fontsize=13)
    axes[1].legend(handles=[Patch(color=cmap(i), label=l) for i, l in enumerate(
        ["1등급: 기존 대여소 상위 20% 수준", "2등급", "3등급", "4등급", "5등급: 하위 20% 수준"])] +
        [plt.Line2D([], [], marker="o", ls="", color="black", ms=3, label="기존 대여소")],
        frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9.5)
    fig.suptitle("신규 대여소 후보지 예상 수요 등급 — 기존 대여소 200m~1.5km 범위의 250m 격자 (검토 우선순위 참고용)",
                 x=0.01, ha="left", fontsize=14)
    save(fig, "S4_candidate_grade_map.png")
    t = pd.DataFrame(rows)
    to_csv(t, "S4_grade_counts.csv")
    to_csv(pd.concat(tops), "S4_top20_candidates.csv")
    say(t.to_string(index=False))
    by_gu = pd.concat([pd.read_csv(OUT / f"S4_candidates_{c}.csv") for c in ("daejeon", "changwon")])
    gu = by_gu.groupby(["city", "gu"]).agg(후보수=("등급", "size"), 일등급=("등급", lambda s: (s == 1).sum()),
                                          일이등급=("등급", lambda s: (s <= 2).sum())).reset_index()
    to_csv(gu, "S4_grade_by_gu.csv")
    say(gu.to_string(index=False))
    return t


def main():
    font()
    OUT.mkdir(parents=True, exist_ok=True)
    say("모델:", type(gbm()).__name__, "| QUICK" if QUICK else "", "| MGWR" if DO_MGWR else "")
    df, gus = load()
    df = residuals(df)
    s1_moran(df, gus)
    s2_gwr(df, gus)
    if DO_MGWR:
        s2_mgwr(df)
    s3_city_regression(df)
    s4_candidates(df, gus)
    (OUT / "summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    say("\n끝. 결과 요약:", OUT / "summary.txt")


if __name__ == "__main__":
    main()
