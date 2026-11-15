"""
따릉이 성공 요인의 타 지역 적용 가능성 — Phase 3: 머신러닝 기반 전이 가능성 검증 (H1~H5)
(가설·판단 기준: docs/hypotheses.md)

E1 [H1]    서울 내부 공간 블록 CV (1·3·5km 민감도) + 인구 단독 기준선
           → 서울 3km CV Spearman이 가장 높은 트리 모델을 '주 모델'로 선택 (타 도시 결과로 고르지 않음)
E2 [H2,H3] 서울 학습 → 대전·세종·창원 예측
           Spearman (vs 인구 기준선, 공간 블록 부트스트랩 95% CI), R2, MAE, RMSE, Bias, 예측/실제 비율
E3 [H3]    평균 보정: Oracle(진단용, 도시 전체 실제 평균) vs Calibration(대여소 10%로 추정 → 나머지 90% 평가, 100회)
E4 [H5]    서울 단독 vs 나머지 도시 전체 vs 서울 표본 수(2,765)에 맞춘 다도시 학습(10회)
E5 [H1,H4] 그룹 단위 Permutation Importance (Spearman 기준)
E6 [H1,H4] SHAP — 서울 모델, 배경 데이터 = 서울 학습 데이터, 모든 도시에 같은 기준 적용 (shap 설치 시)

실행 : 프로젝트 루트에서  python src/train_v2.py
       빠른 동작 확인용   python src/train_v2.py --fast
결과 : results/train_v2/
필요 패키지 : pandas numpy scipy scikit-learn matplotlib  (+ 선택: lightgbm, shap)
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "train_v2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET = "log_y"
DROP = ["city", "lat", "lon", "x", "y"]
CITIES = ["seoul", "daejeon", "sejong", "changwon"]
TARGETS = CITIES[1:]
CITY_KO = {"seoul": "서울", "daejeon": "대전", "sejong": "세종", "changwon": "창원"}
BASELINE_FEAT = "grid_pop_total"
GROUPS = {
    "인구": ["grid_pop_total", "grid_households", "grid_housing"],
    "고용·사업체": ["grid_businesses", "grid_workers"],
    "상업": ["shop_300", "shop_500", "shop_1000", "food_500"],
    "버스": ["bus_300", "bus_500", "bus_1000"],
    "지하철": ["dist_subway_m", "subway_500", "subway_1000"],
    "공원": ["dist_park_m", "park_500", "park_area_500"],
    "대학": ["dist_univ_m", "univ_300", "univ_500", "univ_1000"],
    "경사": ["slope_value"],
}


def set_font():
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR",
                                   "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


# ================================================================ 데이터·모델
def load():
    df = pd.read_csv(DATA / "stations_features_v2.csv", encoding="utf-8-sig")
    cols = pd.read_csv(DATA / "model_feature_columns_v2.csv", encoding="utf-8-sig")["column"].tolist()
    feats = [c for c in cols if c not in DROP]
    for km in (1, 3, 5):
        m = km * 1000
        df[f"block_{km}km"] = (df["city"] + "_" + (df["x"] // m).astype(int).astype(str)
                               + "_" + (df["y"] // m).astype(int).astype(str))
    groups = {g: [f for f in fs if f in feats] for g, fs in GROUPS.items()}
    groups["결측 표시"] = [f for f in feats if f.endswith("_missing")]
    return df, feats, groups


def make_models(feats, fast):
    n = 100 if fast else 500
    num = [c for c in feats if not c.endswith("_missing")]
    flags = [c for c in feats if c.endswith("_missing")]
    linear_pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(lambda a: np.log1p(np.clip(a, 0, None)))),
                          ("sc", StandardScaler())]), num),
        ("flag", "passthrough", flags)])
    m = {
        "Ridge": Pipeline([("pre", linear_pre), ("m", RidgeCV(alphas=np.logspace(-2, 3, 20)))]),
        "RandomForest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("m", RandomForestRegressor(n_estimators=n, min_samples_leaf=5,
                                                              max_features=0.5, n_jobs=-1,
                                                              random_state=SEED))]),
        "HistGB": HistGradientBoostingRegressor(max_iter=n, learning_rate=0.05, min_samples_leaf=20,
                                                l2_regularization=1.0, random_state=SEED),
    }
    if HAS_LGBM:
        m["LightGBM"] = LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=31,
                                      min_child_samples=20, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                                      verbose=-1)
    return m


def sp(a, b):
    return spearmanr(a, b).correlation


def metrics(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(p_log)
    return {
        "Spearman": sp(y_log, p_log),
        "R2_log": r2_score(y_log, p_log),
        "MAE_raw": mean_absolute_error(y, p),
        "RMSE_raw": np.sqrt(mean_squared_error(y, p)),
        "MAE_log": mean_absolute_error(y_log, p_log),
        "RMSE_log": np.sqrt(mean_squared_error(y_log, p_log)),
        "Bias_raw": float(np.mean(p - y)),
        "Ratio_pred_actual": float(np.mean(p) / np.mean(y)),
        "actual_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "n": len(y),
    }


def pop_baseline(d):
    return d[BASELINE_FEAT].fillna(d[BASELINE_FEAT].median()).values


# ================================================================ E1
def e1_spatial_cv(df, feats, models):
    s = df[df.city == "seoul"].reset_index(drop=True)
    X, y = s[feats], s[TARGET].values
    rows, oof3 = [], {}
    for km in (1, 3, 5):
        g = s[f"block_{km}km"].values
        folds = list(GroupKFold(n_splits=5).split(X, y, g))
        for name, mdl in models.items():
            oof = np.zeros(len(s))
            for tr, va in folds:
                m = clone(mdl).fit(X.iloc[tr], y[tr])
                oof[va] = m.predict(X.iloc[va])
            rows.append({"block_km": km, "model": name, **metrics(y, oof)})
            if km == 3:
                oof3[name] = oof
        rows.append({"block_km": km, "model": "Baseline_pop", "Spearman": sp(y, pop_baseline(s)),
                     "n": len(s)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E1_seoul_spatial_cv.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in list(models) + ["Baseline_pop"]:
        r = res[res.model == name]
        ax.plot(r.block_km, r.Spearman, "o-" if name != "Baseline_pop" else "s--", label=name)
    ax.set_xticks([1, 3, 5]); ax.set_xlabel("공간 블록 크기 (km)"); ax.set_ylabel("Spearman")
    ax.set_title("E1 서울 내부 공간 CV — 블록 크기 민감도")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "E1_block_sensitivity.png", dpi=150); plt.close(fig)
    return res, oof3, s


# ================================================================ E2
def block_bootstrap_diff(y, p_model, p_base, blocks, n_boot, rng):
    """공간 블록 단위 부트스트랩: Spearman(모델) - Spearman(기준선)의 95% CI"""
    ub = np.unique(blocks)
    idx_by_b = {b: np.where(blocks == b)[0] for b in ub}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([idx_by_b[b] for b in pick])
        diffs[i] = sp(y[idx], p_model[idx]) - sp(y[idx], p_base[idx])
    return np.nanpercentile(diffs, [2.5, 97.5])


def e2_transfer(df, feats, models, n_boot):
    tr = df[df.city == "seoul"]
    rng = np.random.default_rng(SEED)
    rows, fitted, preds = [], {}, {}
    for name, mdl in models.items():
        m = clone(mdl).fit(tr[feats], tr[TARGET].values)
        fitted[name] = m
        for c in TARGETS:
            te = df[df.city == c]
            y, p = te[TARGET].values, m.predict(te[feats])
            base = pop_baseline(te)
            lo, hi = block_bootstrap_diff(y, p, base, te["block_3km"].values, n_boot, rng)
            rows.append({"model": name, "test_city": c, **metrics(y, p),
                         "Spearman_baseline_pop": sp(y, base),
                         "diff_vs_baseline": sp(y, p) - sp(y, base),
                         "diff_CI95_lo": lo, "diff_CI95_hi": hi,
                         "H2_beats_baseline": bool(lo > 0)})
            preds[(name, c)] = p
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E2_seoul_to_city.csv", index=False, encoding="utf-8-sig")
    return res, fitted, preds


def plot_e2(res, main):
    r = res[res.model == main].set_index("test_city").loc[TARGETS]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    xx = np.arange(len(TARGETS))
    ax[0].bar(xx - .2, r["Spearman"], .4, label=f"서울 모델 ({main})")
    ax[0].bar(xx + .2, r["Spearman_baseline_pop"], .4, label="인구 단독 기준선")
    ax[0].set_xticks(xx, [CITY_KO[c] for c in TARGETS]); ax[0].set_ylabel("Spearman")
    ax[0].set_title("H2 순위 예측: 서울 모델 vs 인구 기준선"); ax[0].legend()
    ax[1].bar(xx - .2, r["actual_mean"], .4, label="실제 평균")
    ax[1].bar(xx + .2, r["pred_mean"], .4, label="예측 평균 (보정 전)")
    for i, c in enumerate(TARGETS):
        ax[1].text(i, max(r.loc[c, "actual_mean"], r.loc[c, "pred_mean"]) + .5,
                   f"비율 {r.loc[c, 'Ratio_pred_actual']:.2f}", ha="center", fontsize=9)
    ax[1].set_xticks(xx, [CITY_KO[c] for c in TARGETS]); ax[1].set_ylabel("건/일")
    ax[1].set_title("H3 절대 수준: 실제 vs 예측 평균"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "E2_rank_vs_level.png", dpi=150); plt.close(fig)


# ================================================================ E3
def rescale(y_ref_log, p_ref_log, p_log):
    """원래 단위(건/일) 평균이 맞도록 배율 보정: k = Σ실제 / Σ예측 (참조 표본에서 추정)
       로그 변환 단위에서 평균을 이동하면 원래 단위 평균이 과보정되므로 배율 방식 사용"""
    k = np.expm1(y_ref_log).sum() / np.expm1(p_ref_log).sum()
    return np.log1p(np.expm1(p_log) * k)


def e3_calibration(df, preds, main, frac=0.10, n_rep=100):
    rng = np.random.default_rng(SEED)
    rows = []
    for c in TARGETS:
        te = df[df.city == c]
        y, p = te[TARGET].values, preds[(main, c)]
        rows.append({"test_city": c, "method": "보정 전", **metrics(y, p)})
        rows.append({"test_city": c, "method": "Oracle(진단용)", **metrics(y, rescale(y, p, p))})
        k = max(5, int(round(frac * len(te))))
        reps_raw, reps_cal = [], []
        for _ in range(n_rep):
            idx = rng.permutation(len(te))
            cal, ev = idx[:k], idx[k:]
            reps_raw.append(metrics(y[ev], p[ev]))
            reps_cal.append(metrics(y[ev], rescale(y[cal], p[cal], p[ev])))
        for label, reps in [("보정 전(평가 90% 동일 표본)", reps_raw),
                            (f"Calibration({int(frac * 100)}%)", reps_cal)]:
            t = pd.DataFrame(reps)
            row = {"test_city": c, "method": label, **t.mean().to_dict()}
            row.update({f"{m}_sd": t[m].std() for m in ["MAE_raw", "RMSE_raw", "Bias_raw", "Ratio_pred_actual"]})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E3_mean_calibration.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    meths = ["보정 전", "Oracle(진단용)", f"Calibration({int(frac * 100)}%)"]
    for ax, met, t in zip(axes, ["MAE_raw", "Bias_raw", "Ratio_pred_actual"],
                          ["MAE (건/일)", "Bias (예측-실제, 건/일)", "예측/실제 평균 비율"]):
        for i, mth in enumerate(meths):
            v = [res[(res.test_city == c) & (res.method == mth)][met].values[0] for c in TARGETS]
            ax.bar(np.arange(3) + (i - 1) * .27, v, .27, label=mth)
        if met == "Ratio_pred_actual":
            ax.axhline(1, color="gray", lw=.8); ax.axhspan(.8, 1.25, color="gray", alpha=.1)
        if met == "Bias_raw":
            ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(3), [CITY_KO[c] for c in TARGETS]); ax.set_title(t)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"E3 도시 평균 보정 효과 (주 모델 {main})")
    fig.tight_layout(); fig.savefig(OUT / "E3_calibration.png", dpi=150); plt.close(fig)
    return res


# ================================================================ E4
def e4_multicity(df, feats, model, n_rep):
    rng = np.random.default_rng(SEED)
    n_seoul = int((df.city == "seoul").sum())
    rows = []
    for c in TARGETS:
        te = df[df.city == c]
        y = te[TARGET].values
        seoul = df[df.city == "seoul"]
        m = clone(model).fit(seoul[feats], seoul[TARGET].values)
        rows.append({"test_city": c, "setting": "① 서울 단독", "n_train": len(seoul), "rep": 0,
                     **metrics(y, m.predict(te[feats]))})
        others = df[df.city != c]
        m = clone(model).fit(others[feats], others[TARGET].values)
        rows.append({"test_city": c, "setting": "② 다도시 전체", "n_train": len(others), "rep": 0,
                     **metrics(y, m.predict(te[feats]))})
        for r in range(n_rep):     # 도시 비율 유지한 층화 추출로 서울 표본 수에 맞춤
            sub = (others.groupby("city", group_keys=False)
                   .apply(lambda g: g.sample(frac=n_seoul / len(others),
                                             random_state=int(rng.integers(1e9)))))
            m = clone(model).fit(sub[feats], sub[TARGET].values)
            rows.append({"test_city": c, "setting": "③ 다도시 동일 표본수", "n_train": len(sub),
                         "rep": r + 1, **metrics(y, m.predict(te[feats]))})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "E4_multicity_raw.csv", index=False, encoding="utf-8-sig")
    keep = ["Spearman", "R2_log", "MAE_raw", "RMSE_raw", "Bias_raw", "Ratio_pred_actual", "n_train"]
    summ = raw.groupby(["test_city", "setting"])[keep].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(OUT / "E4_multicity_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sets = ["① 서울 단독", "② 다도시 전체", "③ 다도시 동일 표본수"]
    for i, st in enumerate(sets):
        r = summ[summ.setting == st].set_index("test_city").loc[TARGETS]
        ax.bar(np.arange(3) + (i - 1) * .27, r["Spearman_mean"], .27,
               yerr=r["Spearman_std"].fillna(0), capsize=3, label=st)
    ax.set_xticks(range(3), [CITY_KO[c] for c in TARGETS]); ax.set_ylabel("Spearman")
    ax.set_title("E4 (H5) 학습 데이터 구성별 전이 성능\n③ vs ① = 도시 다양성 효과, ② vs ③ = 데이터 양 효과")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "E4_multicity.png", dpi=150); plt.close(fig)
    return summ


# ================================================================ E5
def group_pi(model, X, y, groups, n_rep, rng):
    base = sp(y, model.predict(X))
    out = {}
    for g, cols in groups.items():
        if not cols:
            continue
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].values[perm]
            drops.append(base - sp(y, model.predict(Xp)))
        out[g] = np.mean(drops)
    return out


def e5_group_importance(df, feats, groups, model, fitted_main, n_rep):
    rng = np.random.default_rng(SEED)
    s = df[df.city == "seoul"].reset_index(drop=True)
    X, y = s[feats], s[TARGET].values
    acc = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, s["block_3km"].values):
        m = clone(model).fit(X.iloc[tr], y[tr])
        acc.append(group_pi(m, X.iloc[va].reset_index(drop=True), y[va], groups, n_rep, rng))
    res = {"seoul(공간CV)": pd.DataFrame(acc).mean()}
    for c in TARGETS:
        d = df[df.city == c].reset_index(drop=True)
        res[c] = pd.Series(group_pi(fitted_main, d[feats], d[TARGET].values, groups, n_rep, rng))
    imp = pd.DataFrame(res)
    imp = imp.loc[imp["seoul(공간CV)"].sort_values(ascending=False).index]
    imp.to_csv(OUT / "E5_group_permutation_importance.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(imp.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(imp.shape[1]), ["서울(공간CV)"] + [CITY_KO[c] for c in TARGETS])
    ax.set_yticks(range(len(imp)), imp.index)
    for i in range(imp.shape[0]):
        for j in range(imp.shape[1]):
            ax.text(j, i, f"{imp.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman 감소량")
    ax.set_title("E5 그룹 단위 Permutation Importance\n(서울 학습 모델, 순위 예측 기여)")
    fig.tight_layout(); fig.savefig(OUT / "E5_group_importance.png", dpi=150); plt.close(fig)
    return imp


# ================================================================ E6
def e6_shap(df, feats, groups, fitted_main, n_bg, n_explain):
    if not HAS_SHAP:
        log("E6 건너뜀: shap 미설치 (pip install shap)")
        return None
    try:
        model, pre = fitted_main, None
        if isinstance(fitted_main, Pipeline):
            pre, model = fitted_main[:-1], fitted_main[-1]
        tf = (lambda X: pre.transform(X)) if pre is not None else (lambda X: X.values)
        seoul = df[df.city == "seoul"]
        bg = tf(seoul[feats].sample(min(n_bg, len(seoul)), random_state=SEED))
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        rows, dirs = [], []
        for c in CITIES:
            d = df[df.city == c]
            d = d.sample(min(n_explain, len(d)), random_state=SEED)
            Xt = tf(d[feats])
            sv = explainer.shap_values(Xt, check_additivity=False)
            sv = pd.DataFrame(sv, columns=feats)
            for g, cols in groups.items():
                if cols:
                    rows.append({"city": c, "group": g,
                                 "mean_abs_shap": sv[cols].sum(axis=1).abs().mean()})
            for f in feats:
                xv = d[f].values
                ok = ~np.isnan(xv)
                rho = sp(xv[ok], sv[f].values[ok]) if np.unique(xv[ok]).size > 1 else np.nan
                dirs.append({"city": c, "feature": f, "mean_abs_shap": sv[f].abs().mean(),
                             "direction_rho(value,shap)": rho})
        g = pd.DataFrame(rows).pivot(index="group", columns="city", values="mean_abs_shap")[CITIES]
        g = g.loc[g["seoul"].sort_values(ascending=False).index]
        g.to_csv(OUT / "E6_shap_group.csv", encoding="utf-8-sig")
        dd = pd.DataFrame(dirs)
        dd.to_csv(OUT / "E6_shap_feature_direction.csv", index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(8, 5))
        xx = np.arange(len(g))
        for i, c in enumerate(CITIES):
            ax.barh(xx + (i - 1.5) * .2, g[c], .2, label=CITY_KO[c])
        ax.set_yticks(xx, g.index); ax.invert_yaxis()
        ax.set_xlabel("평균 |SHAP| (그룹 합, 로그 변환 단위)")
        ax.set_title("E6 그룹별 SHAP — 서울 모델·서울 배경 데이터 기준")
        ax.legend(); fig.tight_layout(); fig.savefig(OUT / "E6_shap_group.png", dpi=150); plt.close(fig)
        return g
    except Exception as e:  # SHAP 실패가 다른 결과 저장을 막지 않도록
        log(f"E6 SHAP 실패: {type(e).__name__}: {e}")
        return None


# ================================================================ 예측 저장 (Phase 4 지도용)
def save_predictions(df, s_seoul, oof3, preds, main):
    rows = [pd.DataFrame({"city": "seoul", "station": s_seoul["station"].values,
                          "lat": s_seoul["lat"].values, "lon": s_seoul["lon"].values,
                          "actual": s_seoul["daily_mean"].values,
                          "pred": np.expm1(oof3[main]), "pred_oracle": np.expm1(oof3[main]),
                          "source": "서울 공간CV(3km) OOF"})]
    for c in TARGETS:
        te = df[df.city == c]
        p = preds[(main, c)]
        rows.append(pd.DataFrame({"city": c, "station": te["station"].values,
                                  "lat": te["lat"].values, "lon": te["lon"].values,
                                  "actual": te["daily_mean"].values, "pred": np.expm1(p),
                                  "pred_oracle": np.expm1(rescale(te[TARGET].values, p, p)),
                                  "source": "서울 학습 → 전이"}))
    out = pd.concat(rows, ignore_index=True)
    out["gap_oracle"] = out["actual"] - out["pred_oracle"]   # +: 예측보다 잘 됨, -: 저활용
    out["model"] = main
    out.to_csv(OUT / "predictions_main.csv", index=False, encoding="utf-8-sig")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="트리 수·반복 수를 줄여 빠르게 동작 확인")
    args = ap.parse_args()
    f = args.fast
    set_font()
    pd.set_option("display.width", 200)

    df, feats, groups = load()
    models = make_models(feats, f)
    log(f"데이터 {df.shape}, 입력 피처 {len(feats)}개, 모델 {list(models)}"
        + ("" if HAS_LGBM else " (lightgbm 미설치)") + ("" if HAS_SHAP else " (shap 미설치)"))

    log("E1 서울 공간 CV (1·3·5km) ...")
    e1, oof3, s_seoul = e1_spatial_cv(df, feats, models)
    tree = [m for m in models if m != "Ridge"]
    main_name = (e1[(e1.block_km == 3) & e1.model.isin(tree)]
                 .sort_values("Spearman", ascending=False).model.iloc[0])
    log(f"주 모델 = {main_name} (서울 3km CV Spearman 기준)")
    print(e1[["block_km", "model", "Spearman", "R2_log", "MAE_raw", "RMSE_raw"]].round(3).to_string(index=False))

    log("E2 서울 → 타 도시 ...")
    e2, fitted, preds = e2_transfer(df, feats, models, n_boot=200 if f else 1000)
    plot_e2(e2, main_name)
    print(e2[["model", "test_city", "Spearman", "Spearman_baseline_pop", "diff_CI95_lo", "diff_CI95_hi",
              "R2_log", "MAE_raw", "RMSE_raw", "Bias_raw", "Ratio_pred_actual"]].round(3).to_string(index=False))

    log("E3 평균 보정 (Oracle / Calibration) ...")
    e3 = e3_calibration(df, preds, main_name, n_rep=30 if f else 100)
    print(e3[["test_city", "method", "Spearman", "MAE_raw", "RMSE_raw", "Bias_raw",
              "Ratio_pred_actual"]].round(3).to_string(index=False))

    log("E4 다도시 학습 비교 ...")
    e4 = e4_multicity(df, feats, models[main_name], n_rep=3 if f else 10)
    print(e4[["test_city", "setting", "n_train_mean", "Spearman_mean", "Spearman_std",
              "MAE_raw_mean", "Ratio_pred_actual_mean"]].round(3).to_string(index=False))

    log("E5 그룹 Permutation Importance ...")
    e5 = e5_group_importance(df, feats, groups, models[main_name], fitted[main_name], 3 if f else 10)
    print(e5.round(4).to_string())

    log("E6 SHAP ...")
    e6 = e6_shap(df, feats, groups, fitted[main_name], 100 if f else 300, 300 if f else 1000)
    if e6 is not None:
        print(e6.round(4).to_string())

    save_predictions(df, s_seoul, oof3, preds, main_name)
    log(f"저장 완료 → {OUT}")


if __name__ == "__main__":
    main()
