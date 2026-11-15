"""
따릉이 성공 요인의 타 지역 적용 가능성 — Phase 4: 공간 시각화 및 정책 제안 자료

V1 도시별 실제 vs 예측 이용량 지도 + 격차 지도  (기존 대여소, train_v2의 주 모델 예측 사용)
     - 서울: 공간 CV(3km) 예측 / 대전·세종·창원: 서울 학습 모델을 도시 평균 수준만 보정(배율)한 예측
     - 격차 = log1p(실제) - log1p(예측)  → +: 입지 대비 잘 됨,  -: 입지 대비 저활용
V2 저활용·초과 성과 대여소 목록 (도시별 상·하위 20곳)
V3 신규 대여소 후보지 유망도 지도 (대전·세종·창원)
     - 서울 학습 주 모델로 후보지 이용량 예측
     - 유망도 등급: 같은 도시 기존 대여소 예측값 분포 기준 5분위 (1등급 = 기존 대여소 상위 20% 수준)
       → 서울 모델의 '순위'만 사용하므로 도시 간 절대 수준 차이(H3)의 영향을 받지 않음
     - 학습 범위 판정: 같은 도시 기존 대여소의 1~99% 범위를 벗어난 피처 수가
       기존 대여소 95%가 해당하는 기준보다 많거나, SGIS 통계 전체 결측·경사도 결측이면 '범위 밖' (예측 신뢰 낮음)
V4 구별 유망 후보지 요약 (정책 제안용)

실행 : 프로젝트 루트에서  python src/visualization.py
선행 : src/train_v2.py (predictions_main.csv), src/candidate_features.py (candidates_features_v2.csv)
결과 : results/visualization/
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_v2 import load as load_train, make_models, TARGET  # noqa: E402  (같은 모델 설정 재사용)

DATA = ROOT / "data"
OUT = ROOT / "results" / "visualization"
OUT.mkdir(parents=True, exist_ok=True)
PRED = ROOT / "results" / "train_v2" / "predictions_main.csv"
CAND = DATA / "candidates_features_v2.csv"
BOUNDARY = DATA / "raw" / "boundary" / "skorea-municipalities-2018-geo.json"
CITIES = ["seoul", "daejeon", "sejong", "changwon"]
TARGETS = CITIES[1:]
CITY_KO = {"seoul": "서울", "daejeon": "대전", "sejong": "세종", "changwon": "창원"}
CITY_CODE = {"seoul": "11", "daejeon": "25", "sejong": "29", "changwon": "3811"}
GRADE_COLORS = ["#08519c", "#4292c6", "#9ecae1", "#fdd0a2", "#f16913"]   # 1등급(진한 파랑) → 5등급
TOP_N = 20


def set_font():
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR",
                                   "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 경계선
def load_boundaries():
    if not BOUNDARY.exists():
        return {}
    g = json.load(open(BOUNDARY, encoding="utf-8"))
    out = {c: [] for c in CITIES}
    for f in g["features"]:
        code = str(f["properties"]["code"])
        for c, pre in CITY_CODE.items():
            if code.startswith(pre):
                geom = f["geometry"]
                polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
                for poly in polys:
                    ring = np.array(poly[0])
                    out[c].append((f["properties"]["name"].replace("창원시", ""), ring))
    return out


def draw_city(ax, bnd, city, labels=True):
    for name, ring in bnd.get(city, []):
        ax.plot(ring[:, 0], ring[:, 1], color="#555", lw=.6, zorder=1)
    if labels and city != "sejong":
        seen = set()
        for name, ring in bnd.get(city, []):
            if name in seen:
                continue
            seen.add(name)
            ax.text(ring[:, 0].mean(), ring[:, 1].mean(), name, fontsize=8, color="#333",
                    ha="center", va="center", zorder=5, clip_on=True,
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.6))


def fit_view(ax, lon, lat, pad=0.01):
    ax.set_xlim(lon.min() - pad, lon.max() + pad)
    ax.set_ylim(lat.min() - pad, lat.max() + pad)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xticks([]); ax.set_yticks([])


# ================================================================ V1·V2
def v1_v2_maps(P, bnd):
    model = P["model"].iloc[0]
    P = P.copy()
    P["gap_log"] = np.log1p(P["actual"]) - np.log1p(P["pred_oracle"])
    vmin, vmax = np.log1p(P["actual"]).quantile([.02, .98])
    rows = []
    for c in CITIES:
        d = P[P.city == c].sort_values("actual")
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
        for ax, col, title in [(axes[0], "actual", "실제 일평균 대여 건수"),
                               (axes[1], "pred_oracle", "예측 (입지 기반)")]:
            draw_city(ax, bnd, c, labels=False)
            sc = ax.scatter(d.lon, d.lat, c=np.log1p(d[col]), s=7, cmap="viridis", vmin=vmin, vmax=vmax, zorder=3)
            ax.set_title(title); fit_view(ax, d.lon, d.lat)
        cb = fig.colorbar(sc, ax=axes[:2], shrink=.75, pad=.01)
        ticks = [t for t in [1, 3, 10, 30, 100, 300] if vmin <= np.log1p(t) <= vmax]
        cb.set_ticks(np.log1p(ticks)); cb.set_ticklabels([str(t) for t in ticks]); cb.set_label("건/일")

        ax = axes[2]
        draw_city(ax, bnd, c)
        dd = d.reindex(d["gap_log"].abs().sort_values().index)
        lim = max(1.0, float(P["gap_log"].abs().quantile(.98)))
        sc2 = ax.scatter(dd.lon, dd.lat, c=dd["gap_log"], s=8, cmap="RdBu",
                         norm=TwoSlopeNorm(0, -lim, lim), zorder=3)
        ax.set_title("격차 = 실제 − 예측 (로그 변환 단위)\n파랑: 입지 대비 잘 됨 / 빨강: 입지 대비 저활용")
        fit_view(ax, d.lon, d.lat)
        fig.colorbar(sc2, ax=ax, shrink=.75, pad=.01)
        src = d["source"].iloc[0]
        fig.suptitle(f"V1 {CITY_KO[c]} — 실제 vs 예측 이용량 ({model}, {src}"
                     + (", 도시 평균 수준 보정)" if c != "seoul" else ")"), fontsize=13)
        fig.savefig(OUT / f"V1_{c}_actual_pred_gap.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        under = d.nsmallest(TOP_N, "gap_log").assign(type="저활용(입지 대비 낮음)")
        over = d.nlargest(TOP_N, "gap_log").assign(type="초과 성과(입지 대비 높음)")
        rows.append(pd.concat([under, over]))
    t = pd.concat(rows)[["city", "type", "station", "lat", "lon", "actual", "pred_oracle", "gap_log"]]
    t = t.rename(columns={"actual": "실제_건일", "pred_oracle": "예측_건일"}).round(3)
    t.to_csv(OUT / "V2_under_over_performers.csv", index=False, encoding="utf-8-sig")
    return P


# ================================================================ V3·V4
def applicability(st_c, C_c, feats):
    """같은 도시 기존 대여소 1~99% 범위 밖 피처 수 + 기준(기존 대여소 95% 분위)"""
    num = [f for f in feats if not f.endswith("_missing")]
    lo, hi = st_c[num].quantile(.01), st_c[num].quantile(.99)

    def n_out(X):
        v = X[num]
        return ((v < lo) | (v > hi)).sum(axis=1)                 # NaN은 범위 밖으로 세지 않음

    thr = int(np.ceil(n_out(st_c).quantile(.95)))
    n = n_out(C_c)
    bad = (n > thr) | (C_c["grid_all_missing"] == 1) | (C_c["slope_missing"] == 1)
    return n, thr, np.where(bad, "범위 밖", "범위 안")


def v3_v4_candidates(bnd):
    df, feats, _ = load_train()
    model_name = pd.read_csv(PRED, encoding="utf-8-sig")["model"].iloc[0]
    m = make_models(feats, fast=False)[model_name]
    seoul = df[df.city == "seoul"]
    m.fit(seoul[feats], seoul[TARGET].values)

    C = pd.read_csv(CAND, encoding="utf-8-sig")
    outs, summ, ad_rows = [], [], []
    for c in TARGETS:
        st_c = df[df.city == c]
        C_c = C[C.city == c].copy()
        p_exist = m.predict(st_c[feats])
        p_cand = m.predict(C_c[feats])
        k = st_c["daily_mean"].sum() / np.expm1(p_exist).sum()            # 도시 평균 수준 보정 배율
        C_c["pred_daily_calibrated"] = np.expm1(p_cand) * k
        C_c["pct_vs_existing"] = np.searchsorted(np.sort(p_exist), p_cand) / len(p_exist)
        C_c["grade"] = 5 - np.minimum(4, (C_c["pct_vs_existing"] * 5).astype(int))
        n, thr, status = applicability(st_c, C_c, feats)
        C_c["n_feat_out_of_range"], C_c["applicability"] = n.values, status
        nn = cKDTree(st_c[["x", "y"]].values).query(C_c[["x", "y"]].values)
        C_c["nearest_station"], C_c["dist_nearest_station_m"] = st_c["station"].values[nn[1]], nn[0].round(0)
        outs.append(C_c)
        ad_rows.append({"city": c, "n_candidates": len(C_c), "threshold_n_out": thr,
                        "in_range": int((status == "범위 안").sum()),
                        "out_of_range": int((status == "범위 밖").sum()), "level_ratio_k": k})

        ok = C_c[C_c.applicability == "범위 안"]
        g = ok.groupby("gu").agg(candidates=("grade", "size"),
                                 grade1=("grade", lambda s: int((s == 1).sum())),
                                 grade1_2=("grade", lambda s: int((s <= 2).sum())),
                                 mean_pred_daily=("pred_daily_calibrated", "mean"))
        g["share_grade1_2"] = g["grade1_2"] / g["candidates"]
        summ.append(g.reset_index().assign(city=c))

    A = pd.concat(outs, ignore_index=True)
    A.to_csv(OUT / "V3_candidates_scored.csv", index=False, encoding="utf-8-sig")
    ad = pd.DataFrame(ad_rows)
    ad.to_csv(OUT / "V3_applicability_summary.csv", index=False, encoding="utf-8-sig")
    top = (A[A.applicability == "범위 안"].sort_values(["city", "pct_vs_existing"], ascending=[True, False])
           .groupby("city").head(TOP_N))
    top[["city", "gu", "lat", "lon", "grade", "pct_vs_existing", "pred_daily_calibrated",
         "nearest_station", "dist_nearest_station_m", "grid_pop_total", "shop_500", "slope_value"]] \
        .round(3).to_csv(OUT / "V3_top_candidates.csv", index=False, encoding="utf-8-sig")
    S = pd.concat(summ)[["city", "gu", "candidates", "grade1", "grade1_2", "share_grade1_2", "mean_pred_daily"]]
    S.sort_values(["city", "grade1_2"], ascending=[True, False]).round(3) \
        .to_csv(OUT / "V4_gu_summary.csv", index=False, encoding="utf-8-sig")

    cmap = ListedColormap(GRADE_COLORS)
    norm = BoundaryNorm([.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5)
    st_all = df[df.city.isin(TARGETS)]
    for c in TARGETS:
        a = A[A.city == c]
        s = st_all[st_all.city == c]
        fig, ax = plt.subplots(figsize=(10, 10))
        draw_city(ax, bnd, c)
        out = a[a.applicability == "범위 밖"]
        ax.scatter(out.lon, out.lat, s=9, marker="s", color="#d9d9d9", zorder=2, label="학습 범위 밖 (예측 신뢰 낮음)")
        ok = a[a.applicability == "범위 안"].sort_values("grade", ascending=False)
        sc = ax.scatter(ok.lon, ok.lat, s=11, marker="s", c=ok["grade"], cmap=cmap, norm=norm, zorder=3)
        ax.scatter(s.lon, s.lat, s=3, color="black", zorder=4, label="기존 대여소")
        t = top[top.city == c].head(10)
        ax.scatter(t.lon, t.lat, s=90, facecolors="none", edgecolors="red", lw=1.4, zorder=5, label="유망 Top 10")
        cb = fig.colorbar(sc, ax=ax, shrink=.6, ticks=[1, 2, 3, 4, 5])
        cb.set_label("유망도 등급 (1 = 기존 대여소 상위 20% 수준)")
        fit_view(ax, pd.concat([a.lon, s.lon]), pd.concat([a.lat, s.lat]))
        r = ad.set_index("city").loc[c]
        note = "\n※ 창원은 서울 모델이 인구 단독 기준선보다 낫지 않음(H2) → 참고용" if c == "changwon" else ""
        ax.set_title(f"V3 {CITY_KO[c]} 신규 대여소 후보지 유망도 (250m 격자, 서울 학습 모델)\n"
                     f"학습 범위 안 {int(r.in_range):,}곳 / 범위 밖 {int(r.out_of_range):,}곳{note}")
        ax.legend(loc="lower left", fontsize=9)
        fig.savefig(OUT / f"V3_{c}_candidates.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    return ad, top, S


def main():
    set_font()
    pd.set_option("display.width", 180)
    bnd = load_boundaries()
    P = pd.read_csv(PRED, encoding="utf-8-sig", dtype={"station": str})
    print("V1·V2 실제·예측·격차 지도 ...")
    v1_v2_maps(P, bnd)
    if not CAND.exists():
        print("후보지 파일이 없어 V3·V4 건너뜀 (src/candidate_features.py 먼저 실행)")
        return
    print("V3·V4 후보지 유망도 ...")
    ad, top, S = v3_v4_candidates(bnd)
    print("\n학습 범위 판정:")
    print(ad.round(3).to_string(index=False))
    print("\n도시별 유망 후보지 Top 5:")
    print(top.groupby("city").head(5)[["city", "gu", "grade", "pct_vs_existing", "pred_daily_calibrated",
                                       "dist_nearest_station_m"]].round(2).to_string(index=False))
    print("\n구별 1·2등급 후보지 수 (범위 안):")
    print(S.sort_values(["city", "grade1_2"], ascending=[True, False]).groupby("city").head(3)
          .round(2).to_string(index=False))
    print(f"\n저장 완료 → {OUT}")


if __name__ == "__main__":
    main()
