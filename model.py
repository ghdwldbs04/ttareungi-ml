# -*- coding: utf-8 -*-
"""
모델링: 서울에서 배운 '성공 요인'이 대전·창원에서도 통하는가

    python model.py

입력: data/processed/features_station.csv, data/raw/boundary/*municipalities*.json
출력: outputs/model/*.csv, outputs/figures/model/*.png

실험
  E0  서울 내부 검증 (자치구 단위 GroupKFold — 이웃 대여소끼리 정보가 새지 않게)
  E1  제로샷 전이: 서울 전체로 학습 → 대전·창원 그대로 예측
  E2  현지 보정: 현지 대여소 10/25/50%만 써서 (a) 현지만 (b) 서울+현지 (c) 서울 모델+절편 보정
  E3  도시별 요인 중요도 (permutation importance, SHAP 설치돼 있으면 SHAP도)
  E4  예측 오차 지도 (구 경계 위)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "model"
FIG = ROOT / "outputs" / "figures" / "model"
CITIES = {"seoul": "서울 따릉이", "daejeon": "대전 타슈", "changwon": "창원 누비자"}
COLORS = {"seoul": "#2a6fdb", "daejeon": "#e07b39", "changwon": "#3a9e6b"}
SEED = 42

FEATS = ["bus_300", "bus_500", "dist_subway_m", "subway_1000", "dist_univ_m", "dist_park_m", "park_500",
         "park_area_500", "shop_300", "shop_500", "food_500", "pop_1km", "young_share_1km", "workers_1km",
         "firms_1km", "stations_500", "dist_nn_station_m"]   # 자전거도로(bike_*)는 원천 데이터 누락으로 제외
LABELS = {
    "shop_500": "상가 수 (500m)", "shop_300": "상가 수 (300m)", "food_500": "음식점 수 (500m)",
    "firms_1km": "사업체 수 (1km)", "workers_1km": "종사자 수 (1km)", "pop_1km": "인구 (1km)",
    "young_share_1km": "20~39세 비율", "stations_500": "주변 대여소 수 (500m)",
    "dist_nn_station_m": "가장 가까운 대여소 거리", "dist_subway_m": "지하철역 거리", "subway_1000": "지하철역 수 (1km)",
    "bus_300": "버스정류장 수 (300m)", "bus_500": "버스정류장 수 (500m)", "dist_univ_m": "대학 거리",
    "dist_park_m": "공원 거리", "park_500": "공원 수 (500m)", "park_area_500": "공원 면적 (500m)",
}
EXTRA = ["dist_river_m", "cycleway_m_500", "dist_cycleway_m", "apt_share_1km", "oneperson_share_1km"]
BASE_FEATS = list(FEATS)
LABELS.update({"dist_river_m": "하천 거리", "cycleway_m_500": "자전거도로 길이 (500m)", "dist_cycleway_m": "자전거도로 거리",
               "apt_share_1km": "아파트 비율 (1km)", "oneperson_share_1km": "1인가구 비율 (1km)"})
SHARES = ("young_share_1km", "apt_share_1km", "oneperson_share_1km")


def gbm():
    """LightGBM이 있으면 LightGBM, 없으면 sklearn HistGradientBoosting (같은 계열 알고리즘)"""
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
                                             min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def X_of(d, log=False):
    X = d[FEATS].astype(float).copy()
    if log:
        lf = [f for f in FEATS if f not in SHARES]
        X[lf] = np.log1p(X[lf].clip(lower=0))
    return X.fillna(X.median())


def metrics(y, p):
    return {"RMSE(log)": mean_squared_error(y, p) ** 0.5, "MAE(log)": mean_absolute_error(y, p), "R2": r2_score(y, p),
            "Spearman": spearmanr(y, p).correlation, "평균편향(log, 예측-실제)": float(np.mean(p - y)),
            "R2(편향 보정 후)": r2_score(y, p - np.mean(p - y))}


def font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if n in have:
            plt.rcParams["font.family"] = n
            break
    plt.rcParams.update({"axes.unicode_minus": False, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  그림:", FIG / name)


# ------------------------------------------------------------
def main():
    font()
    OUT.mkdir(parents=True, exist_ok=True)
    global FEATS
    v2 = PROC / "features_station_v2.csv"
    df = pd.read_csv(v2 if v2.exists() else PROC / "features_station.csv", dtype={"station": str})
    FEATS = BASE_FEATS + [f for f in EXTRA if f in df.columns]
    print("피처", len(FEATS), "개 사용", "(추가 피처 포함)" if v2.exists() else "(features_extra.py 안 돌림)")
    try:
        from viz_map import load_gu, assign_gu
        gus = load_gu()
        df = assign_gu(df, gus)
    except Exception as e:
        print("구 경계를 못 읽음 -> 서울 내부 검증은 공간 블록(5km)으로 대체:", e)
        gus = None
        df["gu"] = (df.x // 5000).astype(int).astype(str) + "_" + (df.y // 5000).astype(int).astype(str)
    df["gu"] = df["gu"].fillna("기타")
    print("모델:", type(gbm()).__name__)
    S = df[df.city == "seoul"].reset_index(drop=True)

    # ---------------- E0 서울 내부 검증 ----------------
    print("\n[E0] 서울 내부 검증 (자치구 단위 5-fold)")
    gkf = GroupKFold(n_splits=5)
    oof = {k: np.zeros(len(S)) for k in ("평균", "Ridge", "GBM")}
    for tr, te in gkf.split(S, S.log_y, S.gu):
        oof["평균"][te] = S.log_y.iloc[tr].mean()
        oof["Ridge"][te] = ridge().fit(X_of(S.iloc[tr], True), S.log_y.iloc[tr]).predict(X_of(S.iloc[te], True))
        oof["GBM"][te] = gbm().fit(X_of(S.iloc[tr]), S.log_y.iloc[tr]).predict(X_of(S.iloc[te]))
    e0 = pd.DataFrame({k: metrics(S.log_y, v) for k, v in oof.items()}).T.round(3)
    e0.to_csv(OUT / "E0_seoul_cv.csv", encoding="utf-8-sig")
    print(e0.to_string())
    S["pred_cv"] = oof["GBM"]

    if len(FEATS) > len(BASE_FEATS):
        print("\n[E0b] 추가 피처 효과 (서울, Ridge/GBM × 자치구 단위/무작위 검증)")
        rows, full = [], list(FEATS)
        for set_name, fs in (("기존 17개", BASE_FEATS), (f"추가 포함 {len(full)}개", full)):
            FEATS = fs
            for cv_name, splits in (("자치구 단위", list(GroupKFold(5).split(S, S.log_y, S.gu))),
                                    ("무작위", list(KFold(5, shuffle=True, random_state=SEED).split(S)))):
                for m_name, mk, lg in (("Ridge", ridge, True), ("GBM", gbm, False)):
                    p = np.zeros(len(S))
                    for tr, te in splits:
                        p[te] = mk().fit(X_of(S.iloc[tr], lg), S.log_y.iloc[tr]).predict(X_of(S.iloc[te], lg))
                    mm = metrics(S.log_y.values, p)
                    rows.append({"피처": set_name, "검증": cv_name, "모델": m_name, "R2": mm["R2"], "Spearman": mm["Spearman"]})
        FEATS = full
        e0b = pd.DataFrame(rows).round(3)
        e0b.to_csv(OUT / "E0b_feature_sets.csv", index=False, encoding="utf-8-sig")
        print(e0b.pivot_table(index=["검증", "모델"], columns="피처", values=["R2", "Spearman"]).round(3).to_string())

    # ---------------- E1 제로샷 전이 ----------------
    print("\n[E1] 서울 모델 → 대전·창원 그대로 적용")
    m_gbm = gbm().fit(X_of(S), S.log_y)
    m_rdg = ridge().fit(X_of(S, True), S.log_y)
    rows, preds = [], {}
    for c in ("daejeon", "changwon"):
        T = df[df.city == c]
        for name, p in (("평균(서울)", np.full(len(T), S.log_y.mean())),
                        ("Ridge", m_rdg.predict(X_of(T, True))),
                        ("GBM", m_gbm.predict(X_of(T)))):
            rows.append({"도시": CITIES[c], "모델": name, **metrics(T.log_y.values, p)})
        preds[c] = m_gbm.predict(X_of(T))
    e1 = pd.DataFrame(rows).round(3)
    e1.to_csv(OUT / "E1_zero_shot.csv", index=False, encoding="utf-8-sig")
    print(e1.to_string(index=False))
    # E1b: 도시마다 똑같이 측정된 피처만 쓰면? (OSM 자전거도로는 창원이 거의 미등록 -> 제외)
    if "cycleway_m_500" in FEATS:
        print("\n[E1b] 자전거도로(OSM) 피처를 빼고 전이했을 때")
        full = list(FEATS)
        FEATS = [f for f in full if f not in ("cycleway_m_500", "dist_cycleway_m")]
        m_safe = gbm().fit(X_of(S), S.log_y)
        rows = []
        for c in ("daejeon", "changwon"):
            T = df[df.city == c]
            for name, fs, m in (("전체 피처", full, m_gbm), ("자전거도로 제외", FEATS, m_safe)):
                FEATS = fs
                rows.append({"도시": CITIES[c], "피처": name, **metrics(T.log_y.values, m.predict(X_of(T)))})
            FEATS = [f for f in full if f not in ("cycleway_m_500", "dist_cycleway_m")]
        FEATS = full
        e1b = pd.DataFrame(rows).round(3)
        e1b.to_csv(OUT / "E1b_without_cycleway.csv", index=False, encoding="utf-8-sig")
        print(e1b.to_string(index=False))

    df["pred_zero"] = np.nan
    for c, p in preds.items():
        df.loc[df.city == c, "pred_zero"] = p
    df.loc[df.city == "seoul", "pred_zero"] = S["pred_cv"].values

    # ---------------- E2 현지 보정 학습 곡선 ----------------
    print("\n[E2] 현지 데이터 일부로 보정 (5회 반복 평균)")
    rows = []
    for c in ("daejeon", "changwon"):
        T = df[df.city == c].reset_index(drop=True)
        base = m_gbm.predict(X_of(T))
        for frac in (0.0, 0.1, 0.25, 0.5):
            for rep in range(5):
                rng = np.random.default_rng(SEED + rep)
                idx = rng.permutation(len(T))
                k = int(round(frac * len(T)))
                tr, te = idx[:k], idx[k:]
                y_te = T.log_y.values[te]
                r = {"도시": CITIES[c], "현지비율": frac, "rep": rep}
                r["서울 모델 그대로"] = spearmanr(y_te, base[te]).correlation
                r["서울 모델 그대로_RMSE"] = mean_squared_error(y_te, base[te]) ** 0.5
                if k >= 20:
                    off = np.mean(T.log_y.values[tr] - base[tr])
                    r["서울 모델 + 절편 보정_RMSE"] = mean_squared_error(y_te, base[te] + off) ** 0.5
                    loc = gbm().fit(X_of(T.iloc[tr]), T.log_y.iloc[tr]).predict(X_of(T.iloc[te]))
                    r["현지 데이터만"] = spearmanr(y_te, loc).correlation
                    r["현지 데이터만_RMSE"] = mean_squared_error(y_te, loc) ** 0.5
                    pool = pd.concat([S, T.iloc[tr]])
                    Xp = X_of(pool).assign(**{f"is_{c}": (pool.city == c).astype(float).values})
                    w = np.where(pool.city == c, 5.0, 1.0)
                    mp = gbm().fit(Xp, pool.log_y, sample_weight=w)
                    ppool = mp.predict(X_of(T.iloc[te]).assign(**{f"is_{c}": 1.0}))
                    r["서울+현지 함께"] = spearmanr(y_te, ppool).correlation
                    r["서울+현지 함께_RMSE"] = mean_squared_error(y_te, ppool) ** 0.5
                rows.append(r)
    e2 = pd.DataFrame(rows).groupby(["도시", "현지비율"]).mean(numeric_only=True).drop(columns="rep").round(3)
    e2.to_csv(OUT / "E2_local_calibration.csv", encoding="utf-8-sig")
    print(e2.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    styles = {"서울 모델 + 절편 보정": ("#7a7a7a", "--"), "현지 데이터만": ("#e07b39", "-"), "서울+현지 함께": ("#2a6fdb", "-")}
    for ax, c in zip(axes, ("대전 타슈", "창원 누비자")):
        t = e2.loc[c]
        ax.axhline(t["서울 모델 그대로_RMSE"].iloc[0], color="black", lw=1.2, ls=":", label="서울 모델 그대로")
        for k, (col, ls) in styles.items():
            s = t[f"{k}_RMSE"].dropna()
            ax.plot(s.index * 100, s.values, marker="o", lw=2, color=col, ls=ls, label=k)
        ax.set_title(c, loc="left")
        ax.set_xlabel("학습에 쓴 현지 대여소 비율 (%)")
        ax.set_xticks([10, 25, 50])
    axes[0].set_ylabel("예측 오차 RMSE (log, 낮을수록 좋음)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.suptitle("현지 데이터를 조금만 더해도 서울 모델이 얼마나 나아지나", x=0.01, ha="left", fontsize=13)
    save(fig, "E2_learning_curve.png")

    # E1 그림: 실제 vs 예측
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, c in zip(axes, CITIES):
        d = df[df.city == c]
        ax.scatter(d.log_y, d.pred_zero, s=6, alpha=0.4, color=COLORS[c], linewidths=0)
        lim = [min(d.log_y.min(), d.pred_zero.min()), max(d.log_y.max(), d.pred_zero.max())]
        ax.plot(lim, lim, color="gray", lw=1, ls=":")
        mm = metrics(d.log_y.values, d.pred_zero.values)
        tag = "서울 내부 검증" if c == "seoul" else "서울 모델 그대로"
        ax.set_title(f"{CITIES[c]} ({tag})\nR² {mm['R2']:.2f} · 순위상관 {mm['Spearman']:.2f} · 편향 {mm['평균편향(log, 예측-실제)']:+.2f}",
                     loc="left", fontsize=11)
        ticks = [1, 3, 10, 30, 100]
        ax.set_xticks(np.log1p(ticks), ticks); ax.set_yticks(np.log1p(ticks), ticks)
        ax.set_xlabel("실제 대여소 일평균 대여"); ax.set_ylabel("예측")
    fig.suptitle("서울에서 배운 모델의 예측 — 점선 위면 과대예측", x=0.01, ha="left", fontsize=13)
    save(fig, "E1_actual_vs_pred.png")

    # ---------------- E3 도시별 요인 중요도 ----------------
    print("\n[E3] 도시별 요인 중요도 (permutation, CV 평균)")
    imp = {}
    for c in CITIES:
        T = df[df.city == c].reset_index(drop=True)
        splits = GroupKFold(5).split(T, T.log_y, T.gu) if c == "seoul" else KFold(5, shuffle=True, random_state=SEED).split(T)
        vals = []
        for tr, te in splits:
            m = gbm().fit(X_of(T.iloc[tr]), T.log_y.iloc[tr])
            pi = permutation_importance(m, X_of(T.iloc[te]), T.log_y.iloc[te], n_repeats=5, random_state=SEED, scoring="r2")
            vals.append(pi.importances_mean)
        imp[CITIES[c].split()[0]] = np.mean(vals, axis=0)
    I = pd.DataFrame(imp, index=[LABELS[f] for f in FEATS])
    I.to_csv(OUT / "E3_importance.csv", encoding="utf-8-sig")
    In = I.clip(lower=0) / I.clip(lower=0).sum()  # 도시 안 비중(%)
    In = In.sort_values("서울", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(In) + 1.2))
    im = ax.imshow(In.values * 100, cmap="Reds", aspect="auto", vmin=0)
    ax.set_xticks(range(3), In.columns); ax.set_yticks(range(len(In)), In.index); ax.xaxis.tick_top()
    for i in range(len(In)):
        for j in range(3):
            v = In.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white" if v > 15 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, label="도시 안에서 차지하는 중요도 비중 (%)")
    ax.set_title("도시별 모델이 중요하게 본 요인 (permutation importance)", loc="left", fontsize=12, pad=28)
    save(fig, "E3_importance_by_city.png")
    print((In * 100).round(1).to_string())

    try:
        import shap  # 설치돼 있으면 SHAP 요약도
        for c in CITIES:
            T = df[df.city == c]
            m = gbm().fit(X_of(T), T.log_y)
            sv = shap.TreeExplainer(m).shap_values(X_of(T))
            plt.figure()
            shap.summary_plot(sv, X_of(T).rename(columns=LABELS), show=False, max_display=12)
            plt.title(CITIES[c], loc="left")
            save(plt.gcf(), f"E3_shap_{c}.png")
    except ImportError:
        print("  (shap 미설치 → SHAP 그림 생략. pip install shap 후 다시 실행하면 추가됨)")

    # ---------------- E4 오차 지도 ----------------
    df["resid"] = df.pred_zero - df.log_y
    df[["city", "station", "gu", "lat", "lon", "daily_mean", "log_y", "pred_zero", "resid"]].to_csv(
        OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    if gus is not None:
        from viz_map import draw_gu, outline, extent, frame
        from matplotlib.colors import TwoSlopeNorm
        fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), gridspec_kw={"width_ratios": [1.25, 0.9, 1.1]})
        norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
        for ax, c in zip(axes, CITIES):
            d = df[df.city == c].sort_values("resid", key=np.abs)
            outline(ax, gus[c]); draw_gu(ax, gus[c], fs=8)
            for g in gus[c]:
                from matplotlib.patches import PathPatch
                ax.add_patch(PathPatch(g["path"], facecolor="#f4f4f4", edgecolor="none", zorder=0))
            sc = ax.scatter(d.x, d.y, c=d.resid.clip(-1.5, 1.5), cmap="RdBu_r", norm=norm, s=9, linewidths=0, zorder=4)
            frame(ax, extent(gus[c]))
            tag = "서울 내부 검증" if c == "seoul" else "서울 모델 그대로"
            ax.set_title(f"{CITIES[c]} ({tag})", loc="left", fontsize=13)
        cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.012, pad=0.01)
        cb.set_label("예측 − 실제 (log)  빨강: 과대예측 / 파랑: 과소예측")
        fig.suptitle("서울 모델은 어디서 틀리나 — 대여소별 예측 오차", x=0.01, ha="left", fontsize=15)
        save(fig, "E4_error_map.png")

        gu_err = df.groupby(["city", "gu"]).agg(대여소수=("station", "count"), 평균오차_log=("resid", "mean"),
                                                실제_중앙값=("daily_mean", "median")).reset_index()
        gu_err["예측/실제 배율"] = np.exp(gu_err["평균오차_log"]).round(2)
        gu_err.round(3).to_csv(OUT / "E4_error_by_gu.csv", index=False, encoding="utf-8-sig")
        print("\n[E4] 구별 평균 오차 (배율 >1 이면 과대예측)")
        print(gu_err[gu_err.city != "seoul"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
